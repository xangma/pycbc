"""Compile the archived CPU patches and provide independent reference sums.

The sources are exact snapshots, not NumPy emulations. Build flags match
PyCBC setup.py at the manifest's upstream commit. Build products stay outside
this checkout; a C++ compiler, Cython, setuptools and NumPy are required.
"""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def kernels():
    manifest = json.loads((HERE / 'kernels/manifest.json').read_text())
    labels = list(manifest['sha256'])
    sources = {}
    for label in labels:
        source = (HERE / 'kernels' / (label + '.pyx')).read_bytes()
        assert (hashlib.sha256(source).hexdigest() ==
                manifest['sha256'][label])
        sources[label] = source
    identity = (json.dumps(manifest, sort_keys=True) +
                sys.version + np.__version__)
    key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / ('pycbc-chisq-review-' + key)
    cache.mkdir(exist_ok=True)
    names = {label: 'chisq_review_' + label for label in labels}
    if not (cache / 'complete').exists():
        for label, source in sources.items():
            (cache / (names[label] + '.pyx')).write_bytes(source)
        compiler_args = ['-O3', '-w', '-ffast-math', '-ffinite-math-only']
        linker_args = []
        if platform.machine() == 'x86_64':
            compiler_args.append('-msse4.2')
        if sys.platform == 'darwin':
            compiler_args.append('-stdlib=libc++')
            linker_args.append('-stdlib=libc++')
        else:
            compiler_args.append('-fopenmp')
            linker_args.append('-fopenmp')
        builder = '''from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
names = %r
extensions = [Extension(name, [name + '.pyx'], language='c++',
    include_dirs=[numpy.get_include()], libraries=['m'],
    extra_compile_args=%r, extra_link_args=%r) for name in names]
setup(name='chisq-review', ext_modules=cythonize(extensions, quiet=True))
''' % (list(names.values()), compiler_args, linker_args)
        (cache / 'setup.py').write_text(builder)
        print('Compiling archived CPU kernels once; build log:',
              cache / 'build.log')
        with (cache / 'build.log').open('w') as log:
            run = subprocess.run([sys.executable, 'setup.py', 'build_ext',
                                  '--inplace'], cwd=cache, stdout=log,
                                 stderr=subprocess.STDOUT)
        if run.returncode:
            raise RuntimeError('Kernel build failed; see ' +
                               str(cache / 'build.log'))
        (cache / 'complete').write_text('ok\n')
    modules = {}
    for label, name in names.items():
        library = next(cache.glob(name + '*.so'))
        spec = importlib.util.spec_from_file_location(name, library)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[label] = module
    return modules


def load_capture():
    path = HERE.parent / 'data/point-input-05.npz'
    expected = ('1b824a1b7887634d8fce7e3a7c356972'
                'f417686a2a9924b59413d262592d67ce')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def reference_power(corr, indices, bins):
    """Independent complex128 sum with modular phases, no recurrence."""
    n = len(corr)
    result = np.zeros(len(indices), dtype=np.float64)
    for lo, hi in zip(bins[:-1], bins[1:]):
        frequency = np.arange(int(lo), int(hi), dtype=np.int64)
        values = corr[int(lo):int(hi)].astype(np.complex128)
        for j, point in enumerate(indices):
            # Integers avoid large floating-point phase arguments.
            coordinate = int(point) if point == int(point) else float(point)
            cycles = (frequency * coordinate) % n
            phase = np.exp(2j * np.pi * cycles / n)
            result[j] += abs(np.sum(values * phase, dtype=np.complex128))**2
    return result


def plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True,
                         'grid.alpha': .2, 'lines.linewidth': 1.8})
