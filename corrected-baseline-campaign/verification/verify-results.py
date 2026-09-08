#!/usr/bin/env python3
"""Offline verification of the frozen 2026-09-08 four-arm acquisition.

Exit 0 means the requested evidence scope is consistent, even when scientific_status is FAIL.
Qualification-only mode is explicitly provisional. Dependency metadata mismatch remains disclosed.
Exit 1 means missing, corrupt, inconsistent, or failed acquisition evidence.
No subprocesses, PyCBC imports, benchmark execution, or evidence writes occur.
"""
# Derived from prior verifier SHA256: 61040d988b30236cfcf92696e347d4a04e3f3966310881c6460b7d6972ab514b
import argparse
import base64
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
import tempfile
import types

sys.dont_write_bytecode = True
import numpy as np

FROZEN = {
    'campaign.py': '4c10272822310a6f4e07b9068f014a164d64a420ee4a1778e2316b9993316a2f',
    'config.json': '78d4c5bb2bf45278a98a15689f25f1f1a10efac33f931a22cfe36e0247f4d944',
    'setup-remote.py': 'c4f30a5edd4e1131d601dd266ee2f0bbfe48c538a47f015e286d58e99ceb2f07',
    'checked-inspiral.py': '9181ff0515fb3cdacdd4246166bf90fea0da66232605922040ccc98c63d38b29',
    'compare-triggers.py': 'f0af115a2bf2d3a5a85152cb1f61d9b7570a460efd6d420566c5a2b74ac8f1b5',
    'qualify-inspiral.py': '234ff35cc4d54ab2e9dfd5d475d825a328346183adaf516f46df4653e50450b7',
    'threadpoolctl.py': '12fb9526b6a74d2e686b7ec148dc165c3587999b14fa86984aa180e10802400b',
    'benchmark_cpu_campaign.py': '3ee6008a54da44599665d3a02633fa2328b53d97b81718327b24bb35b9ffff9e',
}
COMMITS = {'corrected': '66789ac4a7468094b0cc3ca1498a1de67e0311f6',
           'proposed': 'f582b6fd250d0b82612492979e01e645d5c07afc'}
EXECUTABLES = {
    'corrected': '5a714f6b5b7c945683837e4287538cb9fc1ed2e3f3d2c93b176df0ebdab9bc3e',
    'proposed': 'd4af378d77aa5f66bcc018db32fe372360e53b22542e539ea3a2e53d31d8fa8a',
}
ROUTES = {'corrected-cpu': ('corrected', 'cpu:1'),
          'proposed-cpu': ('proposed', 'cpu:1'),
          'torch-cpu': ('proposed', 'torch:cpu:1'),
          'torch-cuda': ('proposed', 'torch:cuda:0')}
PAIRS = [('corrected-vs-' + arm, 'corrected-cpu', arm)
         for arm in ('proposed-cpu', 'torch-cpu', 'torch-cuda')] + [
         ('proposed-cpu-vs-' + arm, 'proposed-cpu', arm)
         for arm in ('torch-cpu', 'torch-cuda')]
TOLERANCES = {'rtol': 1e-4, 'atol': 1e-5, 'sigmasq_rtol': 1e-5,
              'phase_atol': 1e-4, 'max_examples': 12}
PSD_RTOL = 1e-4
RESUME_SHA256 = 'f26b846eee2c87836335b3998c4a018681390f515340631249b09b15f7dadbbb'
SOURCE_REVIEW_SHA256 = 'ab4977b38481e4dd133fb3b838a3fc51591c51ca22b43c106f163964101d301a'
DIAGNOSTIC_SCRIPT_SHA256 = 'b98136e61bbcca021250b2fdbde2f59385aaf2d2e279b17e2c806159d2e258fe'
TIMING_SLACK = {
    # Fixed consistency allowances, not measurement uncertainty or revised rates.
    'launch_wait_seconds': 1.0,  # Popen, receipt save, wait polling and scheduling.
    'runtime_boundary_seconds': 5.0,  # Interpreter startup, final receipt and teardown.
    'enclosing_boundary_seconds': 60.0,  # Two 15s GPU calls, 20s join, 10s other monitoring/I/O.
    'wall_clock_seconds': 0.05,  # Timestamp quantization and small wall-clock adjustments.
}
CHECKS = {'one_scalar_controller', 'nonempty_segments',
          'recorded_geometry_matches_controller', 'conditioned_strain_recorded',
          'all_segment_psds_saved', 'all_segment_psds_valid_for_filter',
          'one_compressed_bank', 'all_decompressions_succeeded',
          'all_interpolations_observed', 'no_generation_fallback',
          'bank_0_compression_enabled', 'bank_0_all_templates_once',
          'scalar_ifft_executed_for_every_pair'}


class EvidenceError(ValueError):
    """Evidence cannot support a verdict."""


def require(ok, message):
    if not ok:
        raise EvidenceError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def constant(value):
        raise EvidenceError('Nonfinite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def one(rows, label):
    require(isinstance(rows, list) and len(rows) == 1, label + ': expected one row')
    return rows[0]


def relative(path):
    p = PurePosixPath(path)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == path,
            'Unsafe/noncanonical relative path: ' + str(path))
    return p


def load_frozen(root, name):
    path = root / name
    require(digest(path) == FROZEN[name], 'Frozen helper mismatch: ' + name)
    module = types.ModuleType('_offline_' + name.replace('-', '_').replace('.', '_'))
    module.__file__ = str(path.resolve())
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def gnu_elapsed(text):
    values = re.findall(r'^[ \t]*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):[ \t]*(\S+)[ \t]*$',
                        text, re.MULTILINE)
    require(len(values) == 1, 'GNU time needs exactly one elapsed field')
    value = values[0]
    require(re.fullmatch(r'\d+:\d{2}(?::\d{2})?(?:\.\d+)?', value), 'Invalid GNU time elapsed format')
    fields = value.split(':')
    seconds = float(fields[-1])
    require(seconds < 60 and (len(fields) == 2 or int(fields[1]) < 60), 'GNU time component out of range')
    elapsed = (int(fields[0]) * 60 + seconds if len(fields) == 2 else
               int(fields[0]) * 3600 + int(fields[1]) * 60 + seconds)
    # A full displayed unit covers truncation as well as rounding (including h:mm:ss).
    precision = 10.0 ** -len(value.rsplit('.', 1)[1]) if '.' in value else 1.0
    require(math.isfinite(elapsed) and precision > 0, 'Invalid GNU elapsed precision/range')
    require(re.findall(r'^[ \t]*Exit status:[ \t]*(\d+)[ \t]*$', text, re.MULTILINE) == ['0'],
            'GNU time needs exactly one successful exit status')
    return elapsed, precision


def timing_consistency(receipt, runtime, text):
    """Cross-check the measured process interval against independent recorded clocks."""
    perf = receipt['elapsed_wall_seconds']
    require(type(perf) in (int, float) and math.isfinite(perf) and perf > 0, 'Invalid perf duration')
    gnu, rounding = gnu_elapsed(text)
    wall = TIMING_SLACK['wall_clock_seconds']
    require(-rounding - wall <= perf - gnu <= TIMING_SLACK['launch_wait_seconds'] + rounding,
            'Perf duration disagrees with GNU time')
    outer = [dt.datetime.fromisoformat(receipt[k]) for k in ('started_utc', 'finished_utc')]
    require(all(t.tzinfo is not None for t in outer), 'Enclosing receipt timestamps need timezone')
    start, end = [t.timestamp() for t in outer]
    rs, re_ = [runtime[k] for k in ('started_at', 'finished_at')]
    require(all(type(t) in (int, float) and math.isfinite(t) for t in (rs, re_)) and
            end > start and re_ > rs, 'Invalid receipt/runtime span')
    span = end - start
    runtime_span = re_ - rs
    require(start - wall <= rs < re_ <= end + wall, 'Runtime falls outside enclosing receipt')
    require(-wall <= span - perf <= TIMING_SLACK['enclosing_boundary_seconds'],
            'Perf duration disagrees with enclosing receipt span')
    require(-wall <= perf - runtime_span <= TIMING_SLACK['runtime_boundary_seconds'],
            'Perf duration disagrees with runtime span')
    require(-rounding - wall <= gnu - runtime_span <=
            TIMING_SLACK['runtime_boundary_seconds'] + rounding,
            'GNU duration disagrees with runtime span')
    return dict(perf_seconds=perf, gnu_seconds=gnu, gnu_display_allowance_seconds=rounding,
                runtime_span_seconds=runtime_span, enclosing_span_seconds=span,
                perf_minus_gnu_seconds=perf - gnu, perf_minus_runtime_seconds=perf - runtime_span,
                enclosing_minus_perf_seconds=span - perf, status='PASS')


def dependency_diagnostic(diag, dependencies, timing_status, dependency_hash, status_hash, diagnostic_hash, script_hash):
    """Validate the frozen post-run explanation; never rewrite setup metadata."""
    require(valid_hash(diagnostic_hash) and script_hash == DIAGNOSTIC_SCRIPT_SHA256,
            'Invalid post-run diagnostic artifact identity')
    require(diag['schema_version'] == 1 and
            diag['kind'] == 'post_acquisition_read_only_dependency_diagnostic' and
            diag['dependency_manifest_sha256'] == dependency_hash and
            diag['timing_status_sha256'] == status_hash and
            diag['executable'] == dependencies['executable'] and timing_status['state'] == 'complete',
            'Dependency diagnostic is not bound to this completed acquisition')
    observed = dt.datetime.fromisoformat(diag['observed_at_utc'])
    finished = dt.datetime.fromisoformat(timing_status['finished_utc'])
    require(observed.tzinfo is not None and finished.tzinfo is not None and observed >= finished,
            'Dependency diagnostic must postdate completed timings')
    require(dependencies['packages']['torch'] == diag['metadata_last_wins_torch_version'] == '2.1.1' and
            diag['metadata_first_match_version'] == diag['imported_torch_version'] == '2.13.0+cu130' and
            diag['imported_cuda_version'] == '13.0', 'Unexplained Torch metadata/module discrepancy')
    distributions = diag['distributions']
    require(len(distributions) == 2 and [d['version'] for d in distributions] ==
            [diag['metadata_first_match_version'], diag['metadata_last_wins_torch_version']],
            'Diagnostic does not reproduce first-match/last-wins order')
    roots = []
    selected = []
    for distribution in distributions:
        metadata = PurePosixPath(distribution['metadata_path'])
        require(distribution['name'] == 'torch' and metadata.is_absolute() and
                metadata.name == 'torch-' + distribution['version'] + '.dist-info' and
                set(distribution['metadata_sha256']) == {'METADATA', 'WHEEL', 'RECORD'} and
                all(valid_hash(h) for h in distribution['metadata_sha256'].values()),
                'Incomplete Torch distribution metadata identity')
        roots.append(metadata.parent)
        files = distribution['selected_file_checks']
        require(len(files) == 2 and {f['relative_path'] for f in files} ==
                {'torch/__init__.py', 'torch/version.py'}, 'Missing selected RECORD checks')
        hashes = {}
        for file in files:
            h = file['observed_sha256']
            require(valid_hash(h), 'Malformed selected Torch hash')
            encoded = 'sha256=' + base64.urlsafe_b64encode(bytes.fromhex(h)).decode().rstrip('=')
            require(file['matches_record'] is True and file['recorded'] == encoded and
                    file['path'] == str(metadata.parent / file['relative_path']),
                    'Torch installation selected file disagrees with its RECORD')
            hashes[file['path']] = h
        selected.append(hashes)
    require(roots[0] != roots[1], 'Diagnostic needs distinct Torch installation roots')
    imported = diag['imported_files']
    require(len(imported) == 3 and all(valid_hash(h) for h in imported.values()) and
            all(imported.get(p) == h for p, h in selected[0].items()) and
            all(PurePosixPath(p).parent == roots[0] / 'torch' for p in imported) and
            len([p for p in imported if PurePosixPath(p).name.startswith('_C.') and p.endswith('.so')]) == 1,
            'Imported post-run Torch paths/hashes do not match the first installation')
    return dict(status='RECONCILED_WITH_POST_RUN_DIAGNOSTIC',
        manifest_torch_version=dependencies['packages']['torch'],
        runtime_torch_version=diag['imported_torch_version'], runtime_cuda_version=diag['imported_cuda_version'],
        diagnostic_sha256=diagnostic_hash,
        diagnostic_script_sha256=script_hash,
        manifest_sha256=dependency_hash, timing_status_sha256=status_hash,
        observed_at_utc=diag['observed_at_utc'], hostname=diag['hostname'],
        executable=diag['executable'], original_manifest_unchanged=True,
        explanation='Two distinct installations have matching selected RECORD hashes. Distribution enumeration '
            'collapses duplicate torch names with last value 2.1.1; first-match metadata and the post-run '
            'import identify 2.13.0+cu130. All Torch worker runtime version receipts must match the latter.',
        scope='Worker version receipts and post-run installation identity only. No retrospective hash of '
              'Torch binaries loaded by workers is available; the post-run _C hash does not establish that identity.')


def psd_comparison(a, b):
    """Full-array masks and relative error; no floor or excluded-bin erasure."""
    result = {'shape_equal': a.shape == b.shape, 'dtype_equal': a.dtype == b.dtype,
              'rtol': PSD_RTOL, 'atol': 0, 'mask_equal': {}, 'violations': None}
    if a.shape != b.shape:
        result['pass'] = False
        return result
    masks = {'finite': np.isfinite, 'nan': np.isnan, 'positive_infinity': np.isposinf,
             'negative_infinity': np.isneginf, 'signbit': np.signbit,
             'zero': lambda x: x == 0, 'positive': lambda x: x > 0,
             'negative': lambda x: x < 0}
    result['mask_equal'] = {key: bool(np.array_equal(fn(a), fn(b)))
                            for key, fn in masks.items()}
    finite = np.isfinite(a) & np.isfinite(b)
    x, y = a[finite].astype(np.float64), b[finite].astype(np.float64)
    difference = np.abs(x - y)
    scale = np.maximum(np.abs(x), np.abs(y))
    require(np.isfinite(difference).all() and np.isfinite(scale).all(),
            'PSD comparison arithmetic overflow')
    result.update(violations=int(np.count_nonzero(difference > PSD_RTOL * scale)),
                  compared_finite_bins=int(finite.sum()),
                  max_absolute_difference=float(difference.max(initial=0)),
                  max_relative_difference=float(np.divide(
                      difference, scale, out=np.zeros_like(difference),
                      where=scale != 0).max(initial=0)),
                  byte_equal=a.dtype == b.dtype and a.tobytes() == b.tobytes())
    bad_bins = np.flatnonzero(finite)[difference > PSD_RTOL * scale]
    result['first_finite_violation_bin'] = int(bad_bins[0]) if bad_bins.size else None
    result['last_finite_violation_bin'] = int(bad_bins[-1]) if bad_bins.size else None
    result['pass'] = (result['dtype_equal'] and all(result['mask_equal'].values())
                      and result['violations'] == 0 and bool(finite.any()))
    return result


def conditioning(a, b):
    result = {'conditioned_strain_exact': a['strain'] == b['strain'],
              'geometry_exact': a['geometry'] == b['geometry'], 'psds': []}
    for segment in range(5):
        if segment not in a['psds'] or segment not in b['psds']:
            result['psds'].append({'segment': segment, 'pass': False,
                                   'error': 'Missing segment PSD'})
            continue
        x, y = a['psds'][segment], b['psds'][segment]
        row = psd_comparison(x['array'], y['array'])
        row['segment'] = segment
        row['metadata_equal'] = all(x['record'][key] == y['record'][key]
            for key in ('delta_f_hz', 'dyn_range_factor', 'scaling'))
        bounds = [x['record']['validity'][key] for key in ('filter_bin_start', 'filter_bin_stop')]
        other_bounds = [y['record']['validity'][key] for key in ('filter_bin_start', 'filter_bin_stop')]
        row['filter_bins'] = bounds
        row['filter_bounds_equal'] = bounds == other_bounds
        lo, hi = bounds
        if (bounds == other_bounds and x['array'].ndim == y['array'].ndim == 1 and
                0 <= lo < hi <= min(x['array'].size, y['array'].size)):
            row['in_band'] = psd_comparison(x['array'][lo:hi], y['array'][lo:hi])
            row['in_band']['violation_bin_origin'] = lo
        else:
            row['in_band'] = {'pass': False, 'byte_equal': False, 'error': 'Invalid/different filter bounds'}
        row['pass'] = row['pass'] and row['metadata_equal']
        row['pass'] = row['pass'] and row['filter_bounds_equal']
        result['psds'].append(row)
    result['pass'] = (bool(a['strain']) and bool(b['strain']) and
                      result['conditioned_strain_exact'] and result['geometry_exact'] and
                      len(result['psds']) == 5 and all(p['pass'] for p in result['psds']))
    result['full_psd_pass'] = len(result['psds']) == 5 and all(p['pass'] for p in result['psds'])
    result['in_band_psd_budget_pass'] = len(result['psds']) == 5 and all(
        p.get('in_band', {}).get('pass') is True for p in result['psds'])
    result['in_band_psd_exact'] = len(result['psds']) == 5 and all(
        p.get('in_band', {}).get('byte_equal') is True and p.get('metadata_equal') is True
        for p in result['psds'])
    return result


def eligible(trigger, condition, left_timings, right_timings):
    """Explicit completeness avoids all([]) and trigger-only false positives."""
    return (isinstance(trigger, dict) and trigger.get('status') == 'pass' and
            isinstance(condition, dict) and condition.get('pass') is True and
            condition.get('conditioned_strain_exact') is True and
            condition.get('geometry_exact') is True and
            len(condition.get('psds', [])) == 5 and
            all(row.get('pass') is True for row in condition['psds']) and
            len(left_timings) == len(right_timings) == 4 and
            all(row.get('status') == 'pass' for row in left_timings + right_timings))


def legacy_conditioning(a, b):
    """Reproduce acquired diagnostic fields, separately from stricter verdicts."""
    require(len(a['ordered_psds']) == len(b['ordered_psds']),
            'Acquired conditioning comparison could not have completed: PSD count')
    result = {'conditioned_strain_exact': a['strain'] == b['strain'],
              'geometry_exact': a['geometry'] == b['geometry'], 'psds': []}
    for x, y in zip(a['ordered_psds'], b['ordered_psds']):
        av, bv = x['array'], y['array']
        require(av.shape == bv.shape, 'Acquired PSD comparison shape mismatch')
        finite = np.isfinite(av) & np.isfinite(bv)
        d = np.abs(av[finite].astype(float) - bv[finite].astype(float))
        budget = PSD_RTOL * np.maximum(np.abs(av[finite].astype(float)),
                                       np.abs(bv[finite].astype(float)))
        result['psds'].append(dict(exact=av.dtype == bv.dtype and av.tobytes() == bv.tobytes(),
            nonfinite_masks_equal=bool(np.array_equal(np.isposinf(av), np.isposinf(bv))),
            relative_budget=PSD_RTOL, absolute_floor=0,
            violations=int(np.count_nonzero(d > budget)),
            max_absolute_difference=float(d.max(initial=0))))
    return result


def continuation_conditioning(a, b):
    """Recompute the amended controller's diagnostic, without adopting its gate."""
    require(len(a['ordered_psds']) == len(b['ordered_psds']) > 0, 'Continuation PSD count mismatch')
    result = {'conditioned_strain_exact': a['strain'] == b['strain'],
              'geometry_exact': a['geometry'] == b['geometry'], 'psds': []}
    for x, y in zip(a['ordered_psds'], b['ordered_psds']):
        av, bv = x['array'], y['array']
        require(av.shape == bv.shape and av.dtype == bv.dtype, 'Continuation PSD layout mismatch')
        bounds = [x['record']['validity'][k] for k in ('filter_bin_start', 'filter_bin_stop')]
        require(bounds == [y['record']['validity'][k] for k in ('filter_bin_start', 'filter_bin_stop')]
                == [15360, 1048576], 'Continuation PSD bounds mismatch')
        lo, hi = bounds
        finite = np.isfinite(av) & np.isfinite(bv)
        af, bf = av[finite].astype(float), bv[finite].astype(float)
        bad = np.abs(af - bf) > PSD_RTOL * np.maximum(abs(af), abs(bf))
        masks = all(np.array_equal(fn(av), fn(bv)) for fn in (np.isnan, np.isposinf, np.isneginf))
        result['psds'].append(dict(full_psd_budget_pass=bool(masks and not np.any(bad)),
            full_psd_finite_violations=int(np.count_nonzero(bad)), nonfinite_masks_equal=masks,
            used_psd_exact=av[lo:hi].tobytes() == bv[lo:hi].tobytes(), filter_bins=bounds,
            relative_budget=PSD_RTOL, absolute_floor=0))
    return result


def continuation_gate(comparisons):
    """Used only to validate the disclosed timing amendment, never eligibility."""
    names = [name for name, _, _ in PAIRS]
    return all(name in comparisons and comparisons[name]['result']['status'] == 'pass' and
        comparisons[name]['conditioning']['conditioned_strain_exact'] is True and
        comparisons[name]['conditioning']['geometry_exact'] is True and
        comparisons[name]['conditioning']['in_band_psd_exact'] is True for name in names)


class Verifier:
    def __init__(self, root, qualification_only=False):
        self.root = root.resolve(strict=True)
        self.qualification_only = qualification_only
        self.inventory = {}
        self.receipts = {}
        self.loaded = {}
        self.qualified = {}
        self.archived_files = {'verified': [], 'not_transferred': []}
        self.continuation = None
        self.source_review = None
        self.dependency_versions = {}
        self.timing_checks = {}
        self.dependency_reconciliation = None

    def file(self, rel):
        relative(str(rel))
        path = self.root / rel
        require(path.resolve(strict=True).is_relative_to(self.root), 'Evidence symlink escapes root')
        require(path.is_file(), 'Missing evidence file: ' + str(rel))
        value = digest(path)
        require(str(rel) not in self.inventory or self.inventory[str(rel)] == value,
                'Evidence changed during verification: ' + str(rel))
        self.inventory[str(rel)] = value
        return path

    def read(self, rel):
        return strict_json(self.file(rel).read_text())

    def optional_archived(self, rel, expected):
        path = self.root / rel
        if path.exists() or path.is_symlink():
            require(digest(self.file(rel)) == expected, 'Archived file hash mismatch: ' + rel)
            self.archived_files['verified'].append(rel)
        else:
            self.archived_files['not_transferred'].append(rel)

    def initialize(self):
        for name, expected in FROZEN.items():
            require(digest(self.file(name)) == expected, 'Frozen acquisition file mismatch: ' + name)
        self.config = self.read('config.json')
        require(self.config['source_commits'] == COMMITS, 'Unexpected source revisions')
        self.pins = self.read('source-pins.json')
        require(self.inventory['source-pins.json'] == '3745c9d356dbe2c162f7f57c681179e4f9119d001a269cb57b2089086ccaa219',
                'Source pins differ from reviewed qualification snapshot')
        require(set(self.pins) == set(COMMITS), 'Source pin set mismatch')
        harness = self.read('harness-pins.json')
        require(set(FROZEN).issubset(harness), 'Missing frozen harness pins')
        require('dependencies.json' in harness, 'Missing dependency pin')
        for name, expected in harness.items():
            require(valid_hash(expected) and digest(self.file(name)) == expected,
                    'Acquisition harness pin mismatch: ' + name)
        self.deps = self.read('dependencies.json')
        require(self.deps['packages'] and self.deps['python'] and
                PurePosixPath(self.deps['executable']).is_absolute(), 'Incomplete dependency manifest')
        setup = self.read('setup-status.json')
        require(setup['state'] == 'complete' and setup['completed'] == list(COMMITS),
                'Setup did not complete both sources')
        roots = []
        for name, commit in COMMITS.items():
            pin = self.pins[name]
            require(pin['info'] == dict(commit=commit, status='', tracked_diff=''),
                    'Dirty/unexpected source pin: ' + name)
            require(pin['tracked']['bin/pycbc_inspiral'] == EXECUTABLES[name],
                    'Executable pin differs from independently frozen revision: ' + name)
            build = self.read(name + '-build.json')
            source = PurePosixPath(build['source'])
            require(source.is_absolute() and source.name == name, 'Invalid recorded source path')
            roots.append(source.parent)
            require(build['commit'] == commit and
                    build['native'] == pin['native'] and len(pin['native']) >= 11 and
                    build['version_sha256'] == pin['generated_version'], 'Build/source pins disagree')
            references = {
                'corrected': ('original', '40e94792b3edf59f39b18b65102b28a4f74433a7',
                    '6c319b700a7673184eee2a91940054bec55945facc5e4458d39336ae409da451'),
                'proposed': ('proposed', '123e1fb3ef1b338cada636e71c3e9c7987002402',
                    '3d6592ea1bfc023cdf993f1f13121b1e505ee317ef3698b4b0bef698f2ae2bd6')}
            reference_name, reference_commit, reference_hash = references[name]
            require(build['method'] == 'Reused frozen binaries after exact native-source and binary-hash verification'
                    and build['reference_commit'] == reference_commit
                    and build['reference_build_sha256'] == reference_hash
                    and build['reference_source'] == '/home/xangma/pycbc-torch-baseline-final-20260908/' + reference_name,
                    'Reused native build provenance mismatch')
            for path, value in self.module_pins(name).items():
                relative(path)
                require(valid_hash(value), 'Malformed source content hash')
                self.optional_archived(name + '/' + path, value)
        require(roots[0] == roots[1], 'Source roots differ')
        self.recorded_root = roots[0]
        # Source bytes need not be transferred: report this limitation explicitly.
        for path, value in self.config['input_pins'].items():
            require(valid_hash(value), 'Malformed scientific input hash')
            self.optional_archived('inputs/' + PurePosixPath(path).name, value)
        self.comparator = load_frozen(self.root, 'compare-triggers.py')
        self.qualifier = load_frozen(self.root, 'qualify-inspiral.py')
        self.checker = load_frozen(self.root, 'checked-inspiral.py')
        require(self.comparator.DEFAULTS == TOLERANCES, 'Comparator budgets changed')
        # Adapt the expected location, not the stored runtime snapshots. The helper
        # bytes were checked above; no recorded remote path is opened locally.
        remote_helper = dict(path=str(self.recorded_root / 'threadpoolctl.py'),
                             version='3.6.0', sha256=FROZEN['threadpoolctl.py'])
        self.checker.expected_threadpoolctl = lambda: dict(remote_helper)
        self.status = self.read('status.json')
        self.timing_status = self.status
        if self.status['state'] == 'failed':
            self.validate_continuation()
        else:
            require(self.status['state'] == 'complete', 'Acquisition is incomplete')
            require(not (self.root / 'timing-status.json').exists(), 'Unexpected extra timing status')
        review_path = Path(__file__).resolve().parent / 'source-review.json'
        review = strict_json(review_path.read_text())
        require(digest(review_path) == SOURCE_REVIEW_SHA256 and review['status'] == 'PASS'
                and review['source_pins_sha256'] == self.inventory['source-pins.json'],
                'Independent source review differs from frozen receipt')
        self.source_review = dict(sha256=SOURCE_REVIEW_SHA256, receipt=review)
        diagnostic_names = ('torch-environment-diagnostic.json', 'inspect-torch-environment.py')
        if not self.qualification_only and any((self.root / name).exists() for name in diagnostic_names):
            require(digest(self.file(diagnostic_names[1])) == DIAGNOSTIC_SCRIPT_SHA256,
                    'Unreviewed post-run dependency diagnostic script')
            require(self.continuation is not None, 'Diagnostic requires the completed timing continuation')
            self.dependency_reconciliation = dependency_diagnostic(
                self.read(diagnostic_names[0]), self.deps, self.timing_status,
                self.inventory['dependencies.json'], self.inventory['timing-status.json'],
                self.inventory[diagnostic_names[0]], DIAGNOSTIC_SCRIPT_SHA256)
        self.read('machine.json')

    def validate_continuation(self):
        require(self.status['completed'] == ['qual-' + arm for arm in ROUTES],
                'Continuation may only follow the preserved four-qualification stop')
        require(self.status['error'].startswith('AssertionError(') and 'violations' in self.status['error'],
                'Unrecognized acquisition failure; continuation cannot excuse runtime failures')
        require(digest(self.file('resume-timings.py')) == RESUME_SHA256, 'Unreviewed continuation script')
        pins = self.read('continuation-pins.json')
        require(set(pins) == {'resume-timings.py', 'continuation-policy.json', 'status.json'},
                'Unexpected continuation pin set')
        for name, expected in pins.items():
            require(valid_hash(expected) and digest(self.file(name)) == expected,
                    'Continuation evidence changed: ' + name)
        policy = self.read('continuation-policy.json')
        require(set(policy) == {'purpose', 'initial_stop', 'initial_status_sha256', 'change',
                'post_qualification_amendment', 'full_psd_failures_reclassified_as_pass',
                'acquisition_script_sha256', 'continuation_script_sha256', 'decided_at_utc'},
                'Unexpected continuation policy schema')
        require(policy['initial_stop'] == self.status and
                policy['initial_status_sha256'] == self.inventory['status.json'] and
                policy['acquisition_script_sha256'] == FROZEN['campaign.py'] and
                policy['continuation_script_sha256'] == RESUME_SHA256 and
                policy['post_qualification_amendment'] is True and
                policy['full_psd_failures_reclassified_as_pass'] is False and
                policy['purpose'] == 'Complete descriptive fixed-workload timings; full-PSD failures remain failures.' and
                policy['change'] == 'After observing excluded-bin PSD differences, allow timings only after '
                'all corrected-baseline and proposed-backend trigger comparisons, exact conditioned strain/geometry, and exact used PSD bins pass. '
                'Preserve full-array checks and prohibit full-equivalence speedup claims. No source, input, '
                'tolerance, worker command or ordering changes.', 'Continuation policy does not account for stop')
        decided = dt.datetime.fromisoformat(policy['decided_at_utc'])
        require(decided.tzinfo is not None, 'Continuation policy needs an absolute timestamp')
        if self.qualification_only:
            self.timing_status = None
            self.continuation = dict(policy=policy, pins=pins, timing_status=None,
                                     initial_scientific_failure_preserved=True)
            return
        self.timing_status = self.read('timing-status.json')
        require(self.timing_status['state'] == 'complete' and
                self.timing_status['qualification_status'] == 'status.json' and
                self.timing_status['continuation_policy'] == 'continuation-policy.json' and
                self.timing_status['lock'] == self.status['lock'], 'Timing continuation did not complete correctly')
        self.continuation = dict(policy=policy, pins=pins, timing_status=self.timing_status,
                                 initial_scientific_failure_preserved=True)

    def module_pins(self, source):
        pin = self.pins[source]
        merged = dict(pin['tracked'])
        for path, value in pin['native'].items():
            require(path not in merged or merged[path] == value, 'Conflicting native source pin')
            merged[path] = value
        require('pycbc/version.py' not in merged or
                merged['pycbc/version.py'] == pin['generated_version'], 'Conflicting generated version pin')
        merged['pycbc/version.py'] = pin['generated_version']
        return merged

    def modules(self, observed, source):
        require(isinstance(observed, dict) and observed, 'Missing imported module hashes')
        pins = self.module_pins(source)
        prefix = self.recorded_root / source
        seen = set()
        for path, value in observed.items():
            rel = str(PurePosixPath(path).relative_to(prefix))
            relative(rel)
            require(rel in pins and value == pins[rel], 'Imported module does not match source pin: ' + path)
            seen.add(rel)
        require({'pycbc/__init__.py', 'pycbc/version.py', 'pycbc/scheme.py'}.issubset(seen),
                'Imported module receipt lacks core provenance')
        require(any(p.endswith('.so') for p in seen), 'No native module provenance')
        return seen

    def case(self, arm, mode, repeat=None):
        case = 'qual-' + arm if mode == 'qualify' else f'timing-{arm}-r{repeat}'
        rel = 'runs/' + case
        rec = self.read(rel + '/receipt.json')
        source_name, scheme = ROUTES[arm]
        source = self.recorded_root / source_name
        folder = self.recorded_root / rel
        executable = str(source / 'bin/pycbc_inspiral')
        require(all(rec.get(key) == value for key, value in dict(
            case=case, arm=arm, mode=mode, repeat=repeat, scheme=scheme, state='complete',
            returncode=0, cwd=str(self.recorded_root), source=str(source),
            source_info=self.pins[source_name]['info'], source_status_after='').items()),
            'Run identity/state/source mismatch: ' + case)
        cli = [executable, *self.config['common_args'], '--bank-file', self.config['bank'],
               '--processing-scheme', scheme, '--segment-length', '512',
               '--segment-start-pad', '112', '--segment-end-pad', '16',
               '--output', str(folder / 'triggers.hdf')]
        require(rec['executable_cli'] == cli, 'Run scientific argv mismatch: ' + case)
        expected_inputs = dict(self.config['input_pins'], **{executable: EXECUTABLES[source_name],
            str(self.recorded_root / 'config.json'): FROZEN['config.json'],
            str(self.recorded_root / 'checked-inspiral.py'): FROZEN['checked-inspiral.py']})
        require(rec['input_sha256'] == rec['input_sha256_after'] == expected_inputs,
                'Run input/executable hashes mismatch: ' + case)
        env = self.checker.fixed_environment(self.config, source)
        require(rec['environment'] == env, 'Run environment mismatch: ' + case)
        duration = rec['elapsed_wall_seconds']
        require(type(duration) in (int, float) and math.isfinite(duration) and duration > 0,
                'Invalid run duration: ' + case)
        runtime = self.read(rel + '/runtime.json')
        require(runtime['state'] == 'complete' and runtime['source'] == str(source) and
                runtime['imported_source'] == str(source) and runtime['processing_scheme'] == scheme and
                runtime['helper_sha256'] == FROZEN['checked-inspiral.py'] and
                runtime['environment'] == env and runtime['hostname'] == rec['hostname'],
                'Runtime provenance/state mismatch: ' + case)
        lock = runtime['inherited_lock']
        owner = self.status if mode == 'qualify' else self.timing_status
        require(type(lock['fd']) is int and lock['fd'] >= 3 and
                lock['inode'] == owner['lock']['inode'], 'Runtime lock receipt mismatch')
        inner = cli if mode == 'timing' else [str(self.recorded_root / 'qualify-inspiral.py'),
                 '--receipt', str(folder / 'qualification.json'), '--', *cli]
        command = ['/usr/bin/time', '-v', '-o', str(folder / 'time.txt'), 'taskset', '-c', '8',
            self.deps['executable'], str(self.recorded_root / 'checked-inspiral.py'),
            '--receipt', str(folder / 'runtime.json'), '--config', str(self.recorded_root / 'config.json'),
            '--source', str(source), '--scheme', scheme, '--lock-fd', str(lock['fd']), '--', *inner]
        require(rec['command'] == command and runtime['command'] == inner, 'Worker command mismatch')
        for key in ('before_executable', 'at_first_bank', 'after_executable'):
            self.checker.check(runtime[key], env, bank=key == 'at_first_bank', scheme=scheme)
            require(runtime[key]['torch_imported'] is (source_name == 'proposed'),
                    'Torch import state disagrees with the selected scheme')
        seen = self.modules(runtime['source_modules'], source_name)
        if scheme.startswith('torch:'):
            require('pycbc/types/array_torch.py' in seen and runtime['torch_version'] and
                    runtime['torch_cuda_version'], 'Missing Torch module/runtime provenance')
            packages = {k.lower().replace('_', '-'): v for k, v in self.deps['packages'].items()}
            require(runtime['torch_version'] == '2.13.0+cu130' and runtime['torch_cuda_version'] == '13.0',
                    'Worker Torch version differs from the qualification runtime contract')
            if self.dependency_reconciliation is not None:
                require(runtime['hostname'] == self.dependency_reconciliation['hostname'],
                        'Worker host differs from post-run diagnostic host')
            self.dependency_versions[case] = dict(metadata_torch=packages['torch'],
                worker_torch=runtime['torch_version'], worker_cuda=runtime['torch_cuda_version'],
                metadata_matches_runtime=packages['torch'] == runtime['torch_version'])
        else:
            require(runtime['torch_version'] is None and runtime['torch_cuda_version'] is None,
                    'Unexpected CPU wrapper Torch initialization')
        for name in ('triggers.hdf', 'time.txt', 'stdout.log', 'stderr.log',
                     'host-samples.jsonl', 'gpu-before.json', 'gpu-after.json'):
            self.file(rel + '/' + name)
        require(rec['trigger_sha256'] == self.inventory[rel + '/triggers.hdf'],
                'Trigger HDF bytes differ from acquisition receipt: ' + case)
        self.timing_checks[case] = timing_consistency(rec, runtime, self.file(rel + '/time.txt').read_text())
        hdf = self.comparator.load(self.file(rel + '/triggers.hdf'))
        require(not hdf['metadata']['issues'], 'HDF receipt issues: ' + repr(hdf['metadata']['issues']))
        require(hdf['metadata']['source_snapshot'] == self.pins[source_name]['info'] and
                hdf['metadata']['consumed_input_sha256'] == dict(self.config['input_pins'],
                    **{executable: EXECUTABLES[source_name]}), 'Consumed HDF provenance mismatch')
        require(hdf['detectors']['H1']['intervals'] == [[1187007160.0, 1187009064.0]],
                'HDF valid interval mismatch')
        self.receipts[case], self.loaded[case] = rec, hdf
        if mode == 'qualify':
            self.qualification(arm, case)
        return case

    def qualification(self, arm, case):
        rec = self.receipts[case]
        q = self.read('runs/' + case + '/qualification.json')
        require(q['status'] == 'success' and q['executable_exit_code'] == 0 and
                set(q['checks']) == CHECKS and all(v is True for v in q['checks'].values()),
                'Missing/failed qualification checks: ' + case)
        source = ROUTES[arm][0]
        require(q['source_root'] == rec['source'] and q['argv'] == q['executed_argv'] == rec['executable_cli'] and
                q['executable']['path'] == rec['executable_cli'][0] and
                q['executable']['sha256'] == EXECUTABLES[source] and
                q['wrapper']['sha256'] == FROZEN['qualify-inspiral.py'] and
                q['wrapper']['path'] == str(self.recorded_root / 'qualify-inspiral.py') and
                q['python'] == self.deps['executable'], 'Qualification provenance mismatch')
        self.modules({row['path']: row['sha256'] for row in q['source_modules'].values()}, source)
        obs = q['observations']
        bank = one(obs['banks'], 'Bank')
        require(bank['file']['path'] == self.config['bank'] and
                bank['file']['sha256'] == self.config['input_pins'][self.config['bank']] and
                bank['selected_template_count'] == bank['expected_decompressions'] == 384 and
                bank['expected_filter_calls'] == 1920 and bank['has_compressed_waveforms'] is True and
                bank['enable_compressed_waveforms'] is True, 'Qualification bank/workload mismatch')
        require(set(bank['templates']) == {str(i) for i in range(384)}, 'Missing template work records')
        hashes = set()
        for key, row in bank['templates'].items():
            require(row['index'] == int(key) and
                    all(row[k] == 1 for k in ('getitem_attempts', 'getitem_successes',
                        'decompression_successes', 'expected_decompression_calls')) and
                    row['expected_filter_calls'] == 5 and
                    row['filter_attempts_by_segment'] == row['filter_successes_by_segment'] ==
                    {str(i): 1 for i in range(5)}, 'Incomplete template/segment work')
            hashes.add(row['template_hash'])
        require(len(hashes) == 384, 'Nonunique selected templates')
        calls = obs['decompression_calls']
        require(len(calls) == obs['decompression_success_count'] == 384 and
                {c['index'] for c in calls} == set(range(384)) and
                all(c['bank_id'] == 0 and c['success'] is True and c['actual_interpolation'] for c in calls)
                and obs['waveform_generation_attempts'] == [], 'Decompression/fallback mismatch')
        control = one(obs['matched_filter_controllers'], 'Filter controller')
        require(control['segment_count'] == 5 and control['filter_bin_start'] == 15360 and
                control['filter_bin_stop'] == 1048576, 'Controller geometry mismatch')
        engine = obs['fft_engines'][control['ifft_engine_id']]
        require(engine['nbatch'] == 1 and engine['size'] == 2097152 and
                engine['execute_attempts'] == engine['execute_successes'] == 1920,
                'Incomplete scalar IFFT work')
        require(engine['input_dtype'] == engine['output_dtype'] == 'complex64', 'Unexpected filter FFT precision')
        require(obs['parsed_options']['chisq_bins'] == '16' and
                obs['parsed_options']['cluster_window'] == 1.0, 'Qualification parsed options mismatch')
        geometry = one(obs['segment_geometry'], 'Segment geometry')
        expected = dict(fft_samples=2097152, frequency_samples=1048577, sample_rate_hz=4096,
            delta_f_hz=1 / 512, unique_analyzed_seconds=1904, unique_analyzed_samples=7798784,
            gap_samples=0, overlap_samples=0, bounding_span_seconds=1904,
            union_analyzed_sample_intervals=[[458752, 8257536]])
        require(all(geometry[k] == v for k, v in expected.items()) and
                float(geometry['strain_start_time']) == 1187007048 and len(geometry['segments']) == 5,
                'Fixed 384x1904 geometry mismatch')
        intervals = []
        for row in geometry['segments']:
            seg, ana = row['segment_slice'], row['analyze_slice']
            require(seg['step'] in (None, 1) and ana['step'] in (None, 1) and
                    seg['stop'] - seg['start'] == 2097152 and
                    0 <= ana['start'] < ana['stop'] <= 2097152, 'Invalid actual segment slices')
            pair = [seg['start'] + ana['start'], seg['start'] + ana['stop']]
            require(pair == row['analyzed_sample_interval'], 'Segment interval receipt mismatch')
            intervals.append(pair)
        require(self.qualifier.union_intervals(intervals) == [[458752, 8257536]] and
                sum(b - a for a, b in intervals) == 7798784, 'Actual slices do not cover fixed workload')
        strain = one(obs['conditioned_strain'], 'Conditioned strain')
        require(valid_hash(strain['data_sha256']) and strain['sample_rate_hz'] == 4096 and
                strain['delta_t_seconds'] == 1 / 4096 and
                float(strain['start_time']) == 1187007048 and strain['n_samples'] > 0,
                'Incomplete conditioned strain record')
        dtype = np.dtype(strain['dtype_str'])
        require(dtype.kind == 'f' and dtype.itemsize in (4, 8) and str(dtype) == strain['dtype'] and
                strain['shape'] == [strain['n_samples']] and
                strain['nbytes'] == strain['n_samples'] * dtype.itemsize and
                strain['n_samples'] == 8323072 and float(strain['end_time']) == 1187009080,
                'Conditioned strain layout/extent mismatch')
        gates = strain['gating_info']
        gate_bytes = json.dumps(gates['values'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        require(hashlib.sha256(gate_bytes).hexdigest() == gates['sha256'], 'Conditioned gating hash mismatch')
        saved, per_segment = [], {}
        for row in obs['psd_arrays']:
            relative(row['relative_path'])
            rel = 'runs/' + case + '/' + row['relative_path']
            require(row['path'] == str(self.recorded_root / rel), 'PSD recorded path mismatch')
            path = self.file(rel)
            require(digest(path) == row['sha256'], 'PSD file hash mismatch')
            array = np.load(path, allow_pickle=False)
            require(array.dtype.kind == 'f' and array.dtype.itemsize in (4, 8) and
                    array.shape == (1048577,) and list(array.shape) == row['shape'] and
                    array.dtype.str == row['dtype_str'] and str(array.dtype) == row['dtype'] and
                    array.size == row['n_samples'] and array.nbytes == row['nbytes'] and
                    hashlib.sha256(array.tobytes(order='C')).hexdigest() == row['data_sha256'],
                    'PSD array geometry/dtype/data hash mismatch')
            validity = self.qualifier.psd_validity(array, 15360, 1048576)
            require(row['validity'] == validity and validity['valid_for_filter'] is True and
                    row['finite'] == bool(np.isfinite(array).all()) and row['delta_f_hz'] == 1 / 512 and
                    math.isfinite(row['dyn_range_factor']) and row['dyn_range_factor'] > 0 and
                    row['scaling'] == 'DYN_RANGE_FAC**2', 'Invalid/misreported PSD validity or scaling')
            item = {'array': array, 'record': row}
            saved.append(item)
            require(row['segment_indices'], 'Unassigned PSD')
            for index in row['segment_indices']:
                require(type(index) is int and index in range(5) and index not in per_segment,
                        'Duplicate/invalid PSD segment assignment')
                per_segment[index] = item
        require(set(per_segment) == set(range(5)), 'Missing segment PSD arrays')
        self.qualified[arm] = {'strain': obs['conditioned_strain'], 'geometry': obs['segment_geometry'],
                               'psds': per_segment, 'ordered_psds': saved}
        workload = self.read('runs/' + case + '/workload-check.json')
        require(workload == dict(templates=384, segments=5, valid_seconds=1904, checks=q['checks']),
                'Acquired workload summary mismatch')

    def check_comparison_output(self, actual, archived, left, right, label):
        # Only these two display labels can relocate. Hashes, provenance maps,
        # numerical values, budgets, statuses, counts and all other fields stay exact.
        expected = copy.deepcopy(archived)
        for key, case in (('baseline', left), ('candidate', right)):
            require(expected[key] == str(self.recorded_root / 'runs' / case / 'triggers.hdf'),
                    'Unexpected archived comparison path: ' + label)
            expected[key] = str(self.file('runs/' + case + '/triggers.hdf'))
        require(actual == expected, 'Recomputed comparison differs from archive: ' + label)

    def compare(self, left, right, name):
        a, b = self.loaded[left], self.loaded[right]
        raw = self.comparator.compare(a, b, TOLERANCES)
        raw_archive = self.read('comparisons/' + name + '-raw.json')
        self.check_comparison_output(raw, raw_archive, left, right, name + '-raw')
        candidate = copy.deepcopy(b)
        changes = []
        for key in ('source_snapshot', 'consumed_input_sha256'):
            if candidate['metadata'][key] != a['metadata'][key]:
                changes.append(dict(field=key, before=copy.deepcopy(candidate['metadata'][key]),
                                    after=copy.deepcopy(a['metadata'][key])))
                candidate['metadata'][key] = copy.deepcopy(a['metadata'][key])
        adjusted = self.comparator.compare(a, candidate, TOLERANCES)
        archived = self.read('comparisons/' + name + '.json')
        require(set(archived) == {'result', 'provenance_substitutions', 'tolerances'} and
                archived['provenance_substitutions'] == changes and archived['tolerances'] == TOLERANCES,
                'Unexpected comparison normalization/budgets: ' + name)
        self.check_comparison_output(adjusted, archived['result'], left, right, name)
        return {'raw': raw, 'result': adjusted, 'provenance_substitutions': changes,
                'acquired_raw_and_adjusted_verified': True}

    def compare_timing(self, arm, case):
        result = self.compare('qual-' + arm, case, case + '-vs-own-qualification')
        require(result['result']['status'] == 'pass',
                'Completed timing failed its own qualification comparison: ' + case)
        return result

    def check_case_order(self, cases):
        previous_end = None
        for case in cases:
            rec = self.receipts[case]
            start, end = [dt.datetime.fromisoformat(rec[k]) for k in ('started_utc', 'finished_utc')]
            require(start.tzinfo is not None and end.tzinfo is not None and end > start
                    and (previous_end is None or start >= previous_end), 'Case order/timestamps disagree')
            require(rec['parent_pid'] == self.status['pid'] and rec['hostname'] == 'len',
                    'Qualification host or parent controller mismatch')
            previous_end = end

    def run(self):
        self.initialize()
        order = self.config['ordering']
        arms = list(ROUTES)
        require(self.config['repeats'] == 4 and order == [arms[i:] + arms[:i] for i in range(4)],
                'The frozen four-repeat schedule changed')
        expected_cases = ['qual-' + arm for arm in arms] + [
            f'timing-{arm}-r{repeat}' for repeat, row in enumerate(order, 1) for arm in row]
        if self.qualification_only:
            expected_cases = expected_cases[:4]
        elif self.continuation:
            require(self.timing_status['completed'] == expected_cases[4:],
                    'Incomplete/out-of-order timing continuation status')
        else:
            require(self.status['completed'] == expected_cases, 'Incomplete/out-of-order campaign status')
        actual_cases = {p.name for p in (self.root / 'runs').iterdir() if p.is_dir()}
        if self.qualification_only:
            require(set(expected_cases).issubset(actual_cases), 'Missing qualification case')
            require(all(name in set(expected_cases) or name.startswith('timing-') for name in actual_cases),
                    'Unexpected run directory')
        else:
            require(actual_cases == set(expected_cases), 'Expected exactly four qualifications and 16 timing cases')
        for arm in arms:
            self.case(arm, 'qualify')
        comparisons, acquisition_qual = {}, self.read('qualification-summary.json')
        require(set(acquisition_qual) == {p[0] for p in PAIRS}, 'Missing qualification comparisons')
        for name, left, right in PAIRS:
            row = self.compare('qual-' + left, 'qual-' + right, name)
            self.check_comparison_output(row['result'], acquisition_qual[name],
                                         'qual-' + left, 'qual-' + right, 'qualification-summary:' + name)
            condition = conditioning(self.qualified[left], self.qualified[right])
            require(self.read('comparisons/' + name + '-conditioning.json') ==
                    legacy_conditioning(self.qualified[left], self.qualified[right]),
                    'Acquired conditioning diagnostic mismatch: ' + name)
            comparisons[name] = dict(row, conditioning=condition, left_arm=left, right_arm=right)
        continued_conditions = None
        if self.continuation:
            continued_conditions = {name: continuation_conditioning(self.qualified[left], self.qualified[right])
                                    for name, left, right in PAIRS}
            require(self.read('continuation-conditioning.json') == continued_conditions,
                    'Continuation conditioning diagnostics differ from recomputation')
            require(continuation_gate(comparisons), 'Timing continuation scientific preconditions failed')
            # The frozen initial controller stops at the first failing pair in schedule order.
            failed = next((name for name, left, _ in PAIRS if
                           not comparisons[name]['conditioning']['full_psd_pass']), None)
            require(failed is not None, 'Preserved initial full-PSD failure was not reproduced')
            failure = comparisons[failed]
            corrected_condition = legacy_conditioning(self.qualified[failure['left_arm']],
                                                      self.qualified[failure['right_arm']])
            require(self.status['error'] == repr(AssertionError(corrected_condition)),
                    'Original stop is not the reproduced conditioning assertion')
            self.continuation['reproduced_initial_failure'] = failed
        if self.qualification_only:
            self.check_case_order(expected_cases)
            for rel, value in self.inventory.items():
                require(digest(self.root / rel) == value, 'Evidence changed during verification: ' + rel)
            return dict(evidence_status='PASS_FOR_QUALIFICATION_WITH_LIMITATIONS',
                scientific_status='PASS' if all(v['conditioning']['pass'] and v['result']['status'] == 'pass'
                    for v in comparisons.values()) else 'FAIL',
                equal_output_speedup_eligible=False, timing_status='NOT_EVALUATED_PROVISIONAL_SNAPSHOT',
                comparisons=comparisons, continuation=self.continuation,
                trigger_counts={arm: self.loaded['qual-' + arm]['detectors']['H1']['count'] for arm in arms},
                timing_consistency={'allowances_seconds': TIMING_SLACK, 'cases': self.timing_checks},
                dependency_versions=self.dependency_versions,
                limitations=['Timing acquisition has not been evaluated; no performance verdict is available.',
                    'Setup distribution metadata reports Torch 2.1.1; worker runtime receipts report 2.13.0+cu130. '
                    'This acquisition has no current diagnostic binding that discrepancy to installation bytes.',
                    'Conditioned strain equality uses recorded digests and metadata. Strain and scientific input bytes '
                    'are not transferred; their identity is supported by acquisition receipts.',
                    'Native binary reuse is checked against prior receipts and unchanged source; native bytes are not transferred.'],
                source_content_verification=self.archived_files, independent_source_review_receipt=self.source_review,
                schedule=expected_cases, input_evidence_sha256=dict(sorted(self.inventory.items())))
        timings = {arm: [] for arm in arms}
        within_arm = {arm: [] for arm in arms}
        timing_results = {}
        for repeat, row in enumerate(order, 1):
            for arm in row:
                case = self.case(arm, 'timing', repeat)
                result = self.compare_timing(arm, case)
                timing_results[case] = result
                within_arm[arm].append(result['result'])
                timings[arm].append(self.receipts[case]['elapsed_wall_seconds'])
        previous_end, host = None, None
        for case in expected_cases:
            rec = self.receipts[case]
            start, end = (dt.datetime.fromisoformat(rec[k]) for k in ('started_utc', 'finished_utc'))
            require(start.tzinfo is not None and end.tzinfo is not None and end > start and
                    (previous_end is None or start >= previous_end), 'Receipt time/order inconsistency')
            owner = self.status if case.startswith('qual-') else self.timing_status
            require(rec['parent_pid'] == owner['pid'], 'Wrong campaign parent PID')
            if self.continuation and case.startswith('timing-'):
                require(start >= dt.datetime.fromisoformat(
                    self.continuation['policy']['decided_at_utc']), 'Timing predates continuation policy')
            host = rec['hostname'] if host is None else host
            require(rec['hostname'] == host, 'Cases came from different hosts')
            previous_end = end
        rows = {}
        for arm, samples in timings.items():
            require(len(samples) == 4, 'Missing timing repetition')
            median = statistics.median(samples)
            rows[arm] = dict(source=COMMITS[ROUTES[arm][0]], scheme=ROUTES[arm][1],
                samples_seconds=samples, median_seconds=median, min_seconds=min(samples), max_seconds=max(samples),
                template_seconds_per_wall_second=384 * 1904 / median,
                trigger_count=self.loaded['qual-' + arm]['detectors']['H1']['count'])
        for row in comparisons.values():
            left, right = row['left_arm'], row['right_arm']
            row['eligible'] = eligible(row['result'], row['conditioning'], within_arm[left], within_arm[right])
            row['scientific_status'] = 'PASS' if row['eligible'] else 'FAIL'
            row['descriptive_wall_ratio_left_over_right'] = (
                rows[left]['median_seconds'] / rows[right]['median_seconds'])
        all_eligible = all(row['eligible'] for row in comparisons.values())
        expected_summary = dict(scope=self.config['scope'], timing_boundary=self.config['timing_boundary'],
            arms=rows, tolerances=TOLERANCES,
            corrected_comparisons={k: v['result']['status'] for k, v in comparisons.items() if k.startswith('corrected-')},
            proposed_comparisons={k: v['result']['status'] for k, v in comparisons.items() if k.startswith('proposed-')},
            equal_output_speedup_eligible=all(v['result']['status'] == 'pass' for v in comparisons.values()))
        if self.continuation:
            old_full = all(v['conditioned_strain_exact'] and v['geometry_exact'] and
                           all(p['full_psd_budget_pass'] for p in v['psds'])
                           for v in continued_conditions.values())
            expected_summary['equal_output_speedup_eligible'] &= old_full
            expected_summary.update(full_psd_and_conditioning_pass=old_full,
                                    continuation_policy='continuation-policy.json')
        acquired = self.read('summary.json')
        require(acquired == expected_summary, 'Acquired summary disagrees with its recorded algorithm/receipts')
        differences = []
        if acquired['equal_output_speedup_eligible'] != all_eligible:
            differences.append(dict(field='equal_output_speedup_eligible',
                acquired=acquired['equal_output_speedup_eligible'], authoritative=all_eligible,
                reason='Authoritative eligibility includes conditioning, PSD masks/dtype and within-arm parity'))
        for rel, value in self.inventory.items():
            require(digest(self.root / rel) == value, 'Evidence changed during verification: ' + rel)
        return dict(evidence_status='PASS_WITH_LIMITATIONS', scientific_status='PASS' if all_eligible else 'FAIL',
            equal_output_speedup_eligible=all_eligible,
            proposed_backend_parity=all(comparisons[name]['eligible'] for name, _, _ in PAIRS
                                        if name.startswith('proposed-')),
            comparisons=comparisons, timing_comparisons=timing_results, arms=rows,
            timing_consistency={'allowances_seconds': TIMING_SLACK, 'cases': self.timing_checks},
            dependency_versions=self.dependency_versions,
            dependency_reconciliation=self.dependency_reconciliation,
            dependency_metadata_status=(self.dependency_reconciliation['status'] if self.dependency_reconciliation
                else 'MISMATCH_UNRECONCILED_IN_THIS_ACQUISITION'),
            continuation=self.continuation,
            full_psd_and_conditioning_pass=all(row['conditioning']['pass'] for row in comparisons.values()),
            proposed_in_band_psd_exact=all(comparisons[name]['conditioning']['in_band_psd_exact']
                                          for name, _, _ in PAIRS if name.startswith('proposed-')),
            acquired_summary={'sha256': self.inventory['summary.json'], 'unchanged': True,
                              'recorded_algorithm_verified': True, 'differences': differences},
            schedule=expected_cases, source_content_verification=self.archived_files,
            independent_source_review_receipt=self.source_review,
            provenance_scope='Transferred files were hash-checked where present. Untransferred source/input '
                'bytes are supported only by pinned acquisition receipts; no remote source reconstruction '
                'or live-runtime inspection is claimed. Conditioned strain equality uses recorded digests '
                'and metadata; the strain samples were not stored by acquisition.',
            observed_host=host, recorded_root=str(self.recorded_root),
            input_evidence_sha256=dict(sorted(self.inventory.items())))


def self_test():
    """Focused false-positive and mask/completeness checks; no campaign execution."""
    checks = []
    def check(ok, name):
        require(ok, 'Self-test failed: ' + name)
        checks.append(name)
    def rejects(fn, name):
        try:
            fn()
        except EvidenceError:
            checks.append(name)
        else:
            raise EvidenceError('Self-test accepted: ' + name)
    data = np.array([1.0, 2.0, np.inf], dtype=np.float32)
    item = {'array': data, 'record': dict(delta_f_hz=1 / 512, dyn_range_factor=1.0,
            scaling='DYN_RANGE_FAC**2', validity={'filter_bin_start': 1, 'filter_bin_stop': 2})}
    base = {'strain': [{'data_sha256': 'a' * 64}], 'geometry': [{'fixed': True}],
            'psds': {i: item for i in range(5)}}
    good = conditioning(base, base)
    trigger = {'status': 'pass'}
    repeats = [dict(trigger) for _ in range(4)]
    check(eligible(trigger, good, repeats, repeats), 'complete passing control')
    for label, mutate in (
        ('PSD numerical failure rejects all-trigger-pass', lambda b: b['psds'][0]['array'].__setitem__(0, 1.1)),
        ('strain mismatch rejects all-trigger-pass', lambda b: b['strain'][0].update(data_sha256='b' * 64)),
        ('missing PSD rejects all-trigger-pass', lambda b: b['psds'].pop(0)),
        ('geometry mismatch rejects all-trigger-pass', lambda b: b['geometry'][0].update(fixed=False)),
    ):
        candidate = copy.deepcopy(base)
        mutate(candidate)
        check(not eligible(trigger, conditioning(base, candidate), repeats, repeats), label)
    check(not eligible(trigger, None, repeats, repeats), 'missing conditioning rejects')
    check(not eligible(trigger, good, repeats[:-1], repeats), 'missing timing rejects')
    check(not eligible(trigger, good, repeats, repeats[:3] + [{'status': 'fail'}]),
          'failed timing parity rejects')
    check(not eligible({'status': 'review'}, good, repeats, repeats), 'trigger review never passes')
    check(not psd_comparison(data, data.astype(np.float64))['pass'], 'dtype mismatch rejects')
    check(not psd_comparison(data, np.array([1, 2, 3], dtype=np.float32))['pass'], 'nonfinite mask mismatch rejects')
    check(not psd_comparison(np.array([0.0, 1.0]), np.array([-0.0, 1.0]))['pass'], 'signbit mismatch rejects')
    check(not psd_comparison(np.array([1e-30]), np.array([2e-30]))['pass'], 'no absolute PSD floor')
    check(not psd_comparison(data, data[:2])['pass'], 'shape mismatch rejects')
    excluded = copy.deepcopy(base)
    excluded['psds'][0]['array'][0] = 1.1
    excluded_condition = conditioning(base, excluded)
    check(excluded_condition['in_band_psd_exact'] and
          excluded_condition['in_band_psd_budget_pass'] and not excluded_condition['full_psd_pass'] and
          not eligible(trigger, excluded_condition, repeats, repeats),
          'exact used bins never erase full-PSD scientific failure')
    proposed = {name: dict(result=trigger, conditioning=excluded_condition) for name, _, _ in PAIRS}
    check(continuation_gate(proposed), 'disclosed continuation permits descriptive timings only')
    proposed.pop('proposed-cpu-vs-torch-cuda')
    check(not continuation_gate(proposed), 'missing continuation comparison rejects')
    shifted = copy.deepcopy(base)
    shifted['psds'][0]['record']['validity']['filter_bin_start'] = 0
    check(not conditioning(base, shifted)['pass'], 'different actual filter bounds reject')
    with tempfile.TemporaryDirectory(prefix='selftest-', dir=Path(__file__).resolve().parent) as tmp:
        root = Path(tmp)
        np.save(root / 'psd.npy', data, allow_pickle=False)
        verifier = Verifier(root)
        verifier.file('psd.npy')
        np.save(root / 'psd.npy', data + 1, allow_pickle=False)
        try:
            verifier.file('psd.npy')
        except EvidenceError:
            checks.append('changed evidence bytes reject')
        else:
            raise EvidenceError('Self-test accepted changed evidence')
    # Baseline failures do not invalidate an independently passing proposed pair.
    failed = copy.deepcopy(good)
    failed['pass'] = False
    check(not eligible(trigger, failed, repeats, repeats) and eligible(trigger, good, repeats, repeats),
          'per-comparison eligibility preserves proposed-only ratio')
    verifier = object.__new__(Verifier)
    verifier.compare = lambda *_: {'result': trigger}
    check(verifier.compare_timing('torch-cpu', 'timing-torch-cpu-r1')['result']['status'] == 'pass',
          'completed timing accepts own-qualification PASS')
    for status in ('fail', 'review'):
        verifier.compare = lambda *_, status=status: {'result': {'status': status}}
        rejects(lambda: verifier.compare_timing('torch-cpu', 'timing-torch-cpu-r1'),
                'completed timing rejects own-qualification ' + status + ' as evidence error')
    label = 'Elapsed (wall clock) time (h:mm:ss or m:ss): '
    log = label + '2:00.19\nExit status: 0\n'
    rec = dict(elapsed_wall_seconds=120.2, started_utc='2026-09-08T12:00:00+00:00',
               finished_utc='2026-09-08T12:02:00.6+00:00')
    epoch = dt.datetime.fromisoformat(rec['started_utc']).timestamp()
    runtime = dict(started_at=epoch + .2, finished_at=epoch + 120)
    check(timing_consistency(rec, runtime, log)['status'] == 'PASS', 'consistent independent timing clocks pass')
    check(gnu_elapsed(label + '1:02:03\nExit status: 0\n') == (3723, 1), 'hour-format GNU time has one-second allowance')
    hour_rec = dict(rec, elapsed_wall_seconds=3723.9, finished_utc='2026-09-08T13:02:04.2+00:00')
    check(timing_consistency(hour_rec, dict(started_at=epoch + .2, finished_at=epoch + 3723.5),
                            label + '1:02:03\nExit status: 0\n')['status'] == 'PASS',
          'hour-format truncation allowance passes')
    for bad in ('1:60.01', '1:60:00', '-1:00', 'NaN', '1:00:60'):
        rejects(lambda bad=bad: gnu_elapsed(label + bad + '\nExit status: 0\n'),
                'invalid GNU elapsed rejects: ' + bad)
    rejects(lambda: gnu_elapsed(log + log), 'duplicate GNU time receipt rejects')
    rejects(lambda: gnu_elapsed(log.replace('Exit status: 0', 'Exit status: 1')), 'nonzero GNU exit rejects')
    for duration in (12.02, 1202):
        rejects(lambda duration=duration: timing_consistency(dict(rec, elapsed_wall_seconds=duration), runtime, log),
                'corrupt positive perf duration rejects: ' + str(duration))
    rejects(lambda: timing_consistency(rec, dict(runtime, finished_at=epoch + 60), log),
            'contradictory runtime span rejects')
    rejects(lambda: timing_consistency(rec, dict(runtime, started_at=epoch - 1), log),
            'runtime outside enclosing receipt rejects')
    rejects(lambda: timing_consistency(dict(rec, finished_utc='2026-09-08T12:03:10+00:00'), runtime, log),
            'excess enclosing overhead rejects')

    # Synthetic diagnostic exercises relationships independently of frozen fixture bytes.
    diag = dict(schema_version=1, kind='post_acquisition_read_only_dependency_diagnostic',
        observed_at_utc='2026-09-08T13:00:00+00:00', hostname='fixture-host', executable='/fixture/python',
        dependency_manifest_sha256='a' * 64, timing_status_sha256='b' * 64,
        imported_torch_version='2.13.0+cu130', imported_cuda_version='13.0',
        metadata_first_match_version='2.13.0+cu130', metadata_last_wins_torch_version='2.1.1',
        distributions=[], imported_files={})
    for index, version in enumerate(('2.13.0+cu130', '2.1.1')):
        root = '/fixture/' + str(index) + '/site-packages/'
        h = str(index + 1) * 64
        files = [dict(path=root + name, relative_path=name, observed_sha256=h, matches_record=True,
                      recorded='sha256=' + base64.urlsafe_b64encode(bytes.fromhex(h)).decode().rstrip('='))
                 for name in ('torch/__init__.py', 'torch/version.py')]
        diag['distributions'].append(dict(name='torch', version=version,
            metadata_path=root + 'torch-' + version + '.dist-info',
            metadata_sha256={name: h for name in ('METADATA', 'WHEEL', 'RECORD')}, selected_file_checks=files))
        if index == 0:
            diag['imported_files'] = {f['path']: h for f in files} | {root + 'torch/_C.test.so': h}
    deps = dict(executable='/fixture/python', packages={'torch': '2.1.1'})
    timing = dict(state='complete', finished_utc='2026-09-08T12:30:00+00:00')
    def reconcile(value):
        return dependency_diagnostic(value, deps, timing, 'a' * 64, 'b' * 64, 'd' * 64, DIAGNOSTIC_SCRIPT_SHA256)
    check(reconcile(diag)['original_manifest_unchanged'] and deps['packages']['torch'] == '2.1.1',
          'supported duplicate-installation diagnostic preserves corrected manifest')
    for field, bad in (('dependency_manifest_sha256', 'c' * 64), ('timing_status_sha256', 'c' * 64),
                       ('executable', '/other/python'), ('observed_at_utc', '2026-09-08T12:00:00+00:00'),
                       ('imported_torch_version', '2.1.1'), ('metadata_last_wins_torch_version', '2.13.0+cu130')):
        rejects(lambda field=field, bad=bad: reconcile(dict(diag, **{field: bad})),
                'unbound or contradictory diagnostic rejects: ' + field)
    wrong = copy.deepcopy(diag)
    wrong['distributions'][1]['selected_file_checks'][0]['observed_sha256'] = 'c' * 64
    rejects(lambda: reconcile(wrong), 'selected file must match its recorded checksum')
    wrong = copy.deepcopy(diag)
    wrong['imported_files']['/fixture/0/site-packages/torch/__init__.py'] = 'c' * 64
    rejects(lambda: reconcile(wrong), 'imported hash must match active distribution')
    return dict(status='PASS', checks=checks, count=len(checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path)
    parser.add_argument('--output', type=Path, help='New JSON file outside the evidence root; never overwritten')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--qualification-only', action='store_true', help='Evaluate the four qualifications; never certify timings')
    args = parser.parse_args()
    if args.self_test:
        require(args.evidence_root is None and args.output is None, 'Self-test takes no evidence/output options')
        print(json.dumps(self_test(), indent=2, allow_nan=False))
        return 0
    if args.evidence_root is None or args.output is None:
        parser.error('--evidence-root and --output are required together')
    root, output = args.evidence_root.resolve(), args.output.resolve()
    require(not output.is_relative_to(root), 'Output must be outside the acquired evidence tree')
    require(not output.exists(), 'Refusing to overwrite output')
    require(output.is_relative_to(Path(__file__).resolve().parent), 'Output must remain within the verifier folder')
    report = dict(schema_version=1, verifier_sha256=digest(__file__),
                  verified_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), evidence_root=str(root),
                  frozen_trigger_tolerances=TOLERANCES, psd_tolerances={'rtol': PSD_RTOL, 'atol': 0})
    try:
        report.update(Verifier(root, qualification_only=args.qualification_only).run())
        code = 0
    except Exception as error:
        report.update(evidence_status='FAIL', scientific_status='NOT_EVALUATED',
                      equal_output_speedup_eligible=False, error_type=type(error).__name__, error=str(error))
        code = 1
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(payload)
    print(json.dumps(dict(output=str(output), evidence_status=report['evidence_status'],
                          scientific_status=report['scientific_status'],
                          equal_output_speedup_eligible=report['equal_output_speedup_eligible'])))
    return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (EvidenceError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
