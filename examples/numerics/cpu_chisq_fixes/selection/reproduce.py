"""Repeat the fixed-cut 6144-template H1 comparison in a fresh directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    provenance = json.loads((here/'provenance.json').read_text())
    frame = args.frame.resolve()
    expected = provenance['inputs']['frame_sha256']
    assert hashlib.sha256(frame.read_bytes()).hexdigest() == expected
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inspiral = shutil.which('pycbc_inspiral')
    compressor = shutil.which('pycbc_compress_bank')
    if not inspiral or not compressor:
        raise RuntimeError('Activate the pinned PyCBC environment first')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_DYNAMIC='FALSE',
               MKL_THREADING_LAYER='GNU', NUMEXPR_NUM_THREADS='1',
               PYTHONHASHSEED='0', IMPACT_OUTPUT=str(output))
    env['PYTHONPATH'] = os.pathsep.join([
        str(here.parent), env.get('PYTHONPATH', '')])
    bank = output/'o2-compressed-6144.hdf'
    compression = provenance['banks']['6144']['compression_command'].copy()
    compression[:2] = [sys.executable, compressor]
    compression[compression.index('--bank-file')+1] = str(
        here/'o2-subset-6144.hdf')
    compression[compression.index('--output')+1] = str(bank)
    subprocess.run(compression, env=env, check=True)
    recorded = json.loads((here/'command-6144.json').read_text())['command']
    arguments = recorded[recorded.index('--verbose'):]
    for flag, value in [('--frame-files', frame), ('--bank-file', bank),
                        ('--output', output/'triggers.hdf')]:
        arguments[arguments.index(flag)+1] = str(value)
    command = [sys.executable, '-u', str(here/'observe_arithmetic.py'),
               inspiral, *arguments]
    (output/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    subprocess.run(command, cwd=here.parent, env=env, check=True)


if __name__ == '__main__':
    main()
