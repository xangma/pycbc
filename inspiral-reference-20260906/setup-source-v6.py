#!/usr/bin/env python3
"""Install the reviewed Torch-only hot-path revision with unchanged native code."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

PARENT = '837f38d493420043e45fb1ad210a0ccf68bacbaa'
ALLOWED = [
    'pycbc/fft/torchfft.py', 'pycbc/waveform/decompress_torch.py',
    'test/test_torch_decompress_cpu.py', 'test/test_torch_large_ifft.py',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def main():
    root = Path(__file__).resolve().parent
    version = sys.argv[2] if len(sys.argv) == 3 else 'v6'
    require(version in ('v6', 'v6-candidate1'), 'Unsupported source version')
    old, new = root / 'source-v5', root / ('source-' + version)
    bundle, output = root / ('inspiral-source-' + version + '.bundle'), root / ('source-' + version + '.json')
    commit = sys.argv[1]
    require(len(commit) == 40 and set(commit) <= set('0123456789abcdef'), 'Invalid commit')
    require(not new.exists() and not output.exists(), 'Source or receipt already exists')
    previous = json.loads((root / 'source-v5.json').read_text())
    require(previous['commit'] == git(old, 'rev-parse', 'HEAD') == PARENT, 'Wrong parent')
    require(git(old, 'status', '--porcelain') == '', 'Parent source is dirty')
    inputs = {str(p): digest(p) for p in (Path(__file__), bundle, root / 'source-v5.json')}
    subprocess.run(['git', 'clone', '--no-hardlinks', str(old), str(new)], check=True)
    subprocess.run(['git', '-C', str(new), 'fetch', str(bundle), commit], check=True)
    subprocess.run(['git', '-C', str(new), 'checkout', '--detach', commit], check=True)
    require(git(new, 'rev-parse', 'HEAD^') == PARENT, 'Expected one reviewed hot-path commit')
    changed = git(new, 'diff', '--name-only', PARENT, commit).splitlines()
    require(changed == ALLOWED, f'Unexpected source delta: {changed}')
    native = {}
    for name, expected in previous['native_modules_sha256'].items():
        source, target = old / name, new / name
        require(digest(source) == expected and not target.exists(), f'Unexpected native module: {name}')
        shutil.copy2(source, target)
        native[name] = digest(target)
    require(len(native) == 11 and native == previous['native_modules_sha256'], 'Native code changed')
    require(git(old, 'status', '--porcelain') == git(new, 'status', '--porcelain') == '', 'Dirty source')
    audit = dict(status='pass', source_commit=commit, parent_source_commit=PARENT,
                 dispatch=dict(scheme_prefix='cpu', fft='pycbc.fft.mkl',
                               decompression='pycbc.waveform.decompress_cpu'),
                 rationale='Reviewed cpu:1 selects CPUScheme with cpu dispatch; --fft-backends mkl selects '
                           'the unchanged normal MKL implementation. The schemed compressed-waveform '
                           'entry selects decompress_cpu. The two Torch backend edits add guarded '
                           'execution paths without modifying normal CPU functions or shared dispatch.')
    record = dict(schema_version=1, source=str(new), commit=commit, parent=PARENT,
                  changed_paths=changed, changed_files_sha256={name: digest(new / name) for name in changed},
                  native_modules_sha256=native, bundle_sha256=inputs[str(bundle)],
                  normal_cpu_path_audit=audit,
                  exact_git_diff=git(new, 'diff', '--no-ext-diff', '--no-textconv', PARENT, commit),
                  input_sha256=inputs, input_sha256_after={name: digest(name) for name in inputs})
    require(record['input_sha256'] == record['input_sha256_after'], 'Setup inputs changed')
    # The campaign independently verifies unchanged normal backend/dispatch
    # blobs and AST mappings, plus the frozen tuning and native-module receipts.
    with output.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
