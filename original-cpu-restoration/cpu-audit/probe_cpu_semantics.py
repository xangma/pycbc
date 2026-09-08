"""Run small isolated functions extracted from pinned Git blobs, without PyCBC imports.

These demonstrate Python/NumPy control flow only, not full runtime qualification.
"""
import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

REPO = Path('/Users/xangma/repos/pycbc')
BASE = '40e94792b3edf59f39b18b65102b28a4f74433a7'
CANDIDATE = '652206d84177f658f5fb2f9cab2ee73f6d2bc95f'


def extract(ref, path, symbol, namespace):
    source = subprocess.check_output(
        ['git', '-C', str(REPO), 'show', f'{ref}:{path}'], text=True)
    node = ast.parse(source)
    for name in symbol.split('.'):
        node = next(child for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.ClassDef))
                    and child.name == name)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, f'{ref}:{path}', 'exec'), namespace)
    return namespace[node.name]


def outcome(call):
    try:
        value = call()
        return {'result': 'returned', 'detail': value}
    except Exception as exc:
        return {'result': type(exc).__name__, 'detail': str(exc)}


class Vector:
    ptr = 1

    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


def run(ref):
    result = {}
    array = extract(ref, 'pycbc/types/array.py', 'Array.__array__', {})
    data = np.array([1., 2.], dtype=np.float64)
    converted = array(SimpleNamespace(numpy=lambda: data), dtype=data.dtype)
    result['explicit_same_dtype_aliases'] = bool(np.shares_memory(data, converted))

    check = extract(ref, 'pycbc/fft/core.py', '_check_fwd_args', {})
    result['inplace_r2c_size5'] = {
        str(length): outcome(lambda: check(Vector(length), 'real', Vector(3),
                                          'complex', 1, 5))
        for length in (6, 7)
    }

    namespace = {'numpy': np, '_torch_tensor': lambda _: None}
    if ref == CANDIDATE:
        for name in ('_chisq_zeros', '_chisq_dof_array'):
            extract(ref, 'pycbc/vetoes/chisq.py', name, namespace)
    values = extract(ref, 'pycbc/vetoes/chisq.py',
                     'SingleDetSkyMaxPowerChisq.values', namespace)
    def empty_chisq():
        output = values(SimpleNamespace(do=True, snr_threshold=None),
                        None, None, np.array([], dtype=np.complex64), None,
                        np.array([], dtype=np.int64), None, None,
                        np.array([], dtype=np.float32), 0., 1., 1.)
        return [{'shape': list(item.shape), 'dtype': str(item.dtype)} for item in output]
    result['empty_sky_chisq_no_threshold'] = outcome(empty_chisq)

    namespace = {'numpy': np, '_is_torch_backed': lambda _: False,
                 'sig': SimpleNamespace(windows=SimpleNamespace(hann=np.hanning)),
                 'irfft': np.fft.irfft}
    full_filt = extract(ref, 'pycbc/psd/variation.py', 'create_full_filt', namespace)
    def divide_policy():
        with np.errstate(divide='raise'):
            out = full_filt(np.arange(5, dtype=float), np.ones(5), np.ones(5), 4, 2)
        return {'shape': list(out.shape), 'finite': bool(np.isfinite(out).all())}
    result['psd_variation_divide_raise'] = outcome(divide_policy)
    return result


if __name__ == '__main__':
    report = {'numpy_version': np.__version__, 'kind': 'isolated AST-extracted probes',
              'baseline': {'commit': BASE, 'probes': run(BASE)},
              'candidate': {'commit': CANDIDATE, 'probes': run(CANDIDATE)}}
    output = json.dumps(report, indent=2) + '\n'
    Path(__file__).with_name('probe-results.json').write_text(output)
    print(output)
