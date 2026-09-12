"""Receipt utilities shared by developer qualification tools (not runtime code)."""

import base64
import hashlib
import importlib.util
import importlib.metadata
import importlib.machinery
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np


class DeviceUnavailable(RuntimeError):
    """A requested execution device cannot run; never implies CPU fallback."""


def require_device(requested, torch_module):
    device = torch_module.device(requested)
    if device.type == 'cuda':
        if not torch_module.cuda.is_available():
            raise DeviceUnavailable(f'Requested {requested}: CUDA unavailable')
        if device.index is not None and device.index >= torch_module.cuda.device_count():
            raise DeviceUnavailable(f'Requested {requested}: CUDA index unavailable')
    elif device.type != 'cpu':
        raise DeviceUnavailable(f'Requested {requested}: search qualification supports CPU/CUDA')
    return str(device)


def synchronize(device, torch_module):
    if str(device).startswith('cuda'):
        torch_module.cuda.synchronize(device)


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def search_snapshot(batches, num_templates):
    """Own the result of one drain before another submission reuses buffers."""
    if not isinstance(batches, (list, tuple)) or len(batches) != 1:
        raise ValueError('one completed ticket required per submit/drain')
    batch = batches[0]
    field = (lambda key, default=None: batch.get(key, default)) if isinstance(batch, dict) else (
        lambda key, default=None: getattr(batch, key, default))
    if field('overflow', False) or field('aborted', False):
        raise ValueError('search result overflow or aborted')
    if not field('completed', False):
        raise ValueError('search ticket is not completed')
    rows = field('results')
    if not isinstance(rows, list):
        raise ValueError('search ticket results must be a list')
    columns = {}
    keys = None
    for row in rows:
        if not isinstance(row, dict) or not {'template_id', 'sample_idx', 'snr'} <= row.keys():
            raise ValueError('candidate identity or complex SNR missing')
        if keys is not None and row.keys() != keys:
            raise ValueError('inconsistent candidate fields across tiles')
        keys = row.keys()
        count = len(row['template_id'])
        for key, values in row.items():
            if hasattr(values, 'detach'):
                values = values.detach().cpu().numpy()
            values = np.array(values, copy=True)
            if values.ndim != 1 or len(values) != count or not np.all(np.isfinite(values)):
                raise ValueError(f'invalid or non-finite candidate field: {key}')
            columns.setdefault(key, []).append(values)
    snapshot = {key: np.concatenate(values) for key, values in columns.items()}
    if not snapshot:
        return {'template_id': np.empty(0, np.int64), 'sample_idx': np.empty(0, np.int64),
                'snr': np.empty(0, np.complex64)}
    for name in ('template_id', 'sample_idx'):
        values = snapshot[name]
        if values.dtype.kind not in 'iu' or np.any(values < 0):
            raise ValueError(f'invalid candidate identity field: {name}')
    if np.any(snapshot['template_id'] >= num_templates):
        raise ValueError('candidate template_id outside bank')
    order = np.lexsort((snapshot['sample_idx'], snapshot['template_id']))
    snapshot = {key: value[order].copy() for key, value in snapshot.items()}
    pairs = list(zip(snapshot['template_id'], snapshot['sample_idx']))
    if len(set(pairs)) != len(pairs):
        raise ValueError('duplicate candidate identity')
    return snapshot


def compare_search_snapshots(reference, actual):
    """Compare all exported fields; identities are exact, SNR stays complex."""
    if reference.keys() != actual.keys():
        raise ValueError('candidate output fields differ')
    errors = {}
    for name, expected in reference.items():
        got = actual[name]
        if expected.shape != got.shape:
            raise ValueError(f'candidate count/shape differs: {name}')
        exact = name in {'template_id', 'template_idx', 'sample_idx', 'chisq_dof'}
        error = array_error(expected, got, rtol=0, atol=0)
        if exact:
            passed = np.array_equal(expected, got)
        else:
            atol, rtol = (1e-3, 0.) if name == 'snr' else (1e-6, 1e-6)
            passed = np.allclose(expected, got, rtol=rtol, atol=atol, equal_nan=False)
        if not passed:
            raise ValueError(f'candidate output mismatch: {name}')
        errors[name] = error.get('max_absolute')
    return {'passed': True, 'candidate_count': len(actual['template_id']),
            'max_absolute_errors': errors,
            'output_hashes': {name: array_hash(value) for name, value in actual.items()}}


def binary_identity(path):
    """Record the import/mapping path and actual bytes behind any ABI symlink."""
    observed = Path(path).absolute()
    resolved = observed.resolve()
    result = {'path': str(observed), 'resolved_path': str(resolved)}
    try:
        digest = hashlib.sha256()
        with resolved.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        result.update(sha256=digest.hexdigest(), size_bytes=resolved.stat().st_size)
    except OSError as exc:
        result.update(status='unavailable', reason=str(exc))
    return result


def loaded_implementation_identities():
    """Identify loaded PyCBC extensions and reference waveform/FFT libraries."""
    binaries, reference_versions = {}, {}
    for name, module in list(sys.modules.items()):
        relevant = (name == 'pycbc' or name.startswith('pycbc.') or
                    name.split('.')[0].startswith('lal') or
                    name.startswith(('numpy.', 'scipy.fft', 'pyfftw.')))
        path = getattr(module, '__file__', None)
        if relevant and path and any(path.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
            binaries[name] = binary_identity(path)
        if name in ('lal', 'lalsimulation', 'lalinspiral', 'numpy', 'scipy', 'pyfftw'):
            reference_versions[name] = {'version': getattr(module, '__version__', None),
                                       'path': path}
    backend_module = sys.modules.get('pycbc.fft.backend_cpu')
    backend_name = getattr(backend_module, 'cpu_backend', None)
    implementation = getattr(backend_module, '_adict', {}).get(backend_name)
    fft = {'cpu_backend': backend_name,
           'module': getattr(implementation, '__name__', None),
           'path': getattr(implementation, '__file__', None)}
    # ctypes FFT libraries do not appear as Python extension modules. Linux
    # mappings identify the objects actually loaded, including transitive LAL libs.
    mapped = {}
    maps = Path('/proc/self/maps')
    if maps.is_file():
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith('/'):
                continue
            path = fields[5]
            basename = Path(path).name.lower()
            if path not in mapped and any(part in basename for part in (
                    'liblal', 'libfftw', 'libmkl', 'libopenblas', 'libpocketfft')):
                mapped[path] = binary_identity(path)
    for key in ('double_lib', 'float_lib', '_double_threaded_lib', '_float_threaded_lib', 'lib'):
        library = getattr(implementation, key, None)
        name = getattr(library, '_name', None)
        if isinstance(name, str):
            fft.setdefault('ctypes_libraries', {})[key] = (
                binary_identity(name) if Path(name).is_absolute() else {'loader_name': name})
    return {'loaded_extension_binaries': binaries, 'mapped_reference_libraries': mapped,
            'reference_library_versions': reference_versions, 'reference_fft': fft,
            'mapping_scope': '/proc/self/maps reference-library mappings' if maps.is_file() else
                             'Python extensions and explicit ctypes paths; process mappings unavailable'}


def compare_execution_identity(reference, actual):
    """Require the parent's executed sources and loaded binaries in each child."""
    for key in ('library_versions', 'imported_module_paths'):
        if not reference.get(key) or reference.get(key) != actual.get(key):
            raise ValueError(f'parent/child {key} differs or is missing')
    if not reference.get('repositories') or not actual.get('repositories'):
        raise ValueError('parent/child source provenance missing')
    for name, repo in reference['repositories'].items():
        got = actual.get('repositories', {}).get(name, {})
        if not repo.get('git_commit') or not repo.get('tracked_patch_sha256'):
            raise ValueError(f'parent source identity unavailable: {name}')
        for key in ('root', 'git_commit', 'tracked_patch_sha256', 'untracked_source_base64', 'status'):
            if repo.get(key) != got.get(key):
                raise ValueError(f'parent/child source differs: {name}.{key}')
    for key in ('loaded_extension_binaries', 'mapped_reference_libraries'):
        if key not in reference or key not in actual:
            raise ValueError(f'parent/child binary provenance missing: {key}')
        for name, identity in reference[key].items():
            if identity != actual[key].get(name) or not identity.get('sha256'):
                raise ValueError(f'parent/child binary differs or unavailable: {name}')
    for key in ('reference_fft', 'reference_library_versions'):
        if reference.get(key) != actual.get(key):
            raise ValueError(f'parent/child {key} differs')
    return {'passed': True}


def array_error(reference, actual, rtol, atol=0.0):
    """Unaligned complex errors; use a global norm near sample zeros."""
    ref, got = np.asarray(reference), np.asarray(actual)
    if ref.shape != got.shape:
        return {'passed': False, 'reason': 'shape mismatch',
                'reference_shape': list(ref.shape), 'actual_shape': list(got.shape)}
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(got)):
        return {'passed': False, 'reason': 'non-finite samples'}
    diff = (got.astype(np.complex128) - ref.astype(np.complex128)).ravel()
    scale = float(np.linalg.norm(ref.ravel()))
    error = float(np.linalg.norm(diff))
    return {'passed': bool(error <= atol + rtol * scale),
            'absolute_l2': error, 'reference_l2': scale,
            'relative_l2': error / scale if scale else None,
            'max_absolute': float(np.max(np.abs(diff))) if diff.size else 0.0,
            'rtol': float(rtol), 'atol': float(atol)}


def git_snapshot(root):
    """Store a HEAD-relative binary patch and relevant untracked source bytes.

    Outputs/artifacts are excluded from untracked capture. The scope is explicit;
    imported model assets outside this scope must be supplied as input manifests.
    """
    root = Path(root).resolve()

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root)

    try:
        head = git('rev-parse', 'HEAD').decode().strip()
        patch = git('diff', '--binary', 'HEAD')
        untracked = git('ls-files', '--others', '--exclude-standard', '-z')
        source = {}
        suffixes = {'.py', '.pyx', '.pxd', '.c', '.h', '.cpp', '.cu', '.toml',
                    '.cfg', '.ini', '.yaml', '.yml', '.json', '.npz', '.npy', '.pt'}
        for name in untracked.decode().split('\0'):
            path = root / name
            if not name or name.split('/')[0] in {'artifacts', 'review-input', '.agents'}:
                continue
            if path.suffix in suffixes and path.is_file():
                source[name] = base64.b64encode(path.read_bytes()).decode()
        return {'root': str(root), 'git_commit': head,
                'git_branch': git('rev-parse', '--abbrev-ref', 'HEAD').decode().strip(),
                'git_dirty': bool(git('status', '--porcelain').strip()),
                'tracked_patch_base64': base64.b64encode(patch).decode(),
                'tracked_patch_sha256': hashlib.sha256(patch).hexdigest(),
                'untracked_source_base64': source,
                'untracked_scope': 'code/config/model-asset suffixes; excludes artifacts, review-input, .agents'}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {'root': str(root), 'status': 'unavailable', 'reason': str(exc)}


def execution_provenance(repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    modules = {}
    versions = {}
    repositories = {'pycbc': git_snapshot(root)}
    for name in ('pycbc', 'torchwave', 'torch', 'numpy', 'scipy', 'lalsuite'):
        spec = importlib.util.find_spec(name)
        origin = spec.origin if spec else None
        modules[name] = origin
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        if name == 'torchwave' and origin:
            try:
                tw_root = subprocess.check_output(
                    ['git', '-C', str(Path(origin).parent), 'rev-parse', '--show-toplevel'],
                    stderr=subprocess.DEVNULL).decode().strip()
                repositories[name] = git_snapshot(tw_root)
            except (OSError, subprocess.CalledProcessError):
                repositories[name] = {'status': 'unavailable',
                                      'reason': 'imported distribution has no Git checkout'}
    torch_module = sys.modules.get('torch')
    runtime = {}
    if torch_module is not None:
        runtime = {'torch_num_threads': torch_module.get_num_threads(),
                   'torch_num_interop_threads': torch_module.get_num_interop_threads(),
                   'cuda_runtime': torch_module.version.cuda,
                   'default_dtype': str(torch_module.get_default_dtype()),
                   'float32_matmul_precision': torch_module.get_float32_matmul_precision()}
        if torch_module.cuda.is_available():
            runtime['cuda_devices'] = [
                {'index': i, 'name': torch_module.cuda.get_device_name(i),
                 'total_memory_bytes': torch_module.cuda.get_device_properties(i).total_memory}
                for i in range(torch_module.cuda.device_count())]
    return {**loaded_implementation_identities(),
            'repositories': repositories, 'imported_module_paths': modules,
            'library_versions': versions, 'runtime': runtime,
            'command': [sys.executable, *sys.argv], 'cwd': os.getcwd(),
            'hostname': platform.node(), 'platform': platform.platform(),
            'python': sys.version,
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'thread_environment': {key: os.environ.get(key) for key in
                                   ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                    'CUDA_VISIBLE_DEVICES')},
            'unmeasured_conditions': ['GPU power/clock state', 'other-process contention',
                                      'external model assets outside recorded Git scope']}


def write_receipt(path, receipt):
    """Atomic strict JSON; preserve earlier evidence on serialization failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            out.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def timed_command(command, log_path, cwd=None):
    """Measure subprocess launch through exit, including imports and output."""
    with open(log_path, 'w') as log:
        start = time.perf_counter()
        result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                check=False)
        elapsed = time.perf_counter() - start
    return {'command': command, 'cwd': str(cwd or Path.cwd()),
            'returncode': result.returncode, 'process_wall_time_sec': elapsed,
            'timer_boundary': 'before subprocess launch through process exit; includes output',
            'log_path': str(log_path)}
