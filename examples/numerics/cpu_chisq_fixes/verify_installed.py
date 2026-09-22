"""Record a checkout's public CPU result for comparison with archived kernels.

Run with PYTHONPATH pointing at an editable, built checkout and pass its label.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pycbc
from pycbc.scheme import CPUScheme
from pycbc.types import Array
from pycbc.vetoes import chisq_cpu
from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed

from cpu_chisq_review import load_capture, kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label', choices=[
        'upstream', 'pi', 'shifts', 'arithmetic', 'combined'])
    args = parser.parse_args()
    data = load_capture()
    output = {'label': args.label, 'pycbc_version': pycbc.__version__,
              'package': pycbc.__file__, 'extension': chisq_cpu.__file__,
              'results': {}}
    source = Path(pycbc.__file__).parent / 'vetoes/chisq_cpu.pyx'
    output['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    archived = kernels()[args.label]
    for dtype in [np.complex64, np.complex128]:
        corr = Array(data['corr'].astype(dtype), copy=False)
        with CPUScheme():
            power = chisq_cpu.shift_sum(corr, data['indices'], data['bins'])
            actual = power_chisq_at_points_from_precomputed(
                corr, data['snr'], float(data['norm']),
                data['bins'], data['indices'])
        expected = archived.shift_sum(corr, data['indices'], data['bins'])
        np.testing.assert_array_equal(power, expected)
        # This expression is also used for the patched-kernel public result
        # in the real-data notebook. Verify against the actual Python wrapper.
        wrapper = ((len(data['bins']) - 1)*expected -
                   (data['snr'].conj()*data['snr']).real)
        wrapper *= float(data['norm'])**2
        np.testing.assert_array_equal(actual, wrapper)
        output['results'][np.dtype(dtype).name] = {
            'power': power.tolist(), 'chisq': actual.tolist(),
            'dtype': str(actual.dtype)}
    path = Path(__file__).parent / 'verification'
    path.mkdir(exist_ok=True)
    (path / (args.label + '.json')).write_text(
        json.dumps(output, indent=2)+'\n')
    print(args.label, 'installed and archived results agree exactly')


if __name__ == '__main__':
    main()
