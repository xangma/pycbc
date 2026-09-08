"""Record local PR17 tests and the provenance of reused extensions."""
import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[3]
output = Path(__file__).resolve().parent
previous = Path('/private/tmp/pycbc-original-cpu-followup-validation-20260908-pr17')
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
assert head == '7fd2ce7299ab91e4a4ceab4fd454218c031c3cbe'
verified = []
for pattern in ('*.pyx', '*.pxd', '*.h', '*.so'):
    for old in sorted((previous / 'pycbc').rglob(pattern)):
        relative = old.relative_to(previous)
        old_hash = hashlib.sha256(old.read_bytes()).hexdigest()
        new_hash = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert old_hash == new_hash, relative
        verified.append(dict(path=str(relative), sha256=new_hash))
files = [
    'test_torch_chisq_cpu_compat.py', 'test_torch_chisq_sparse_dispatch.py',
    'test_torch_chisq_cpu_optimization.py', 'test_torch_chisq_precision.py',
    'test_torch_chisq_cuda_normalization.py', 'test_chisq_torch.py',
    'test_cpu_batch_peaks.py', 'test_torch_cpu_native_peaks.py',
    'test_torch_cpu_fft_tuning.py', 'test_torch_large_ifft.py',
    'test_torch_performance_artifacts.py', 'test_hardware.py',
]
command = [sys.executable, '-m', 'pytest', '-q'] + ['test/' + name for name in files]
command += ['--junitxml=' + str(output / 'pytest.xml')]
with (output / 'pytest.log').open('w') as log:
    result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
ruff = '/Users/xangma/miniconda3/bin/ruff'
checks = [
    [ruff, 'check', '--no-cache', 'pycbc/vetoes/chisq_torch.py', '--select', 'F401,B905'],
    [ruff, 'check', '--no-cache', 'test/test_torch_chisq_cpu_optimization.py', '--select', 'F401'],
    [ruff, 'format', '--check', '--no-cache', 'pycbc/vetoes/chisq_torch.py'],
    ['git', 'diff', '--check'],
]
lint = []
with (output / 'lint.log').open('w') as log:
    for check in checks:
        log.write(json.dumps(check) + '\n')
        log.flush()
        completed = subprocess.run(check, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        lint.append(dict(command=check, exit_code=completed.returncode))
suites = ET.parse(output / 'pytest.xml').getroot().findall('testsuite')
counts = {key: sum(int(suite.get(key, '0')) for suite in suites)
          for key in ('tests', 'failures', 'errors', 'skipped')}
counts['passed'] = counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
record = dict(
    created_utc=datetime.now(timezone.utc).isoformat(), head=head, cwd=str(root),
    host=platform.node(), runtime=sys.executable, python=sys.version,
    reused_build_root=str(previous), verified_files=verified,
    pytest_command=command, pytest_exit_code=result.returncode, counts=counts,
    lint=lint, note='CPU/native sources unchanged during conflict resolution. '
    'Three pre-existing B905 warnings in the test file are outside CI F401 scope.',
)
(output / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(head=head, counts=counts, artifact_directory=str(output))))
print((output / 'pytest.log').read_text().splitlines()[-1])
assert result.returncode == 0 and all(item['exit_code'] == 0 for item in lint)
