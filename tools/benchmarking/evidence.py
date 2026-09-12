"""Receipt utilities shared by developer qualification tools (not runtime code)."""

import base64
import hashlib
import importlib.util
import importlib.metadata
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
    for name in ('pycbc', 'torchwave', 'torch', 'numpy'):
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
    return {'repositories': repositories, 'imported_module_paths': modules,
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
