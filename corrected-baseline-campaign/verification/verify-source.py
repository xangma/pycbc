"""Read-only Git/content and reused-native provenance check; emits JSON on stdout."""
import hashlib
import io
import json
import posixpath
from pathlib import Path
import subprocess
import sys
import tarfile

root, repo, previous = map(Path, sys.argv[1:])
pins = json.loads((root / 'source-pins.json').read_text())
commits = {'corrected': '66789ac4a7468094b0cc3ca1498a1de67e0311f6',
           'proposed': 'f582b6fd250d0b82612492979e01e645d5c07afc'}
references = {'corrected': ('original', '40e94792b3edf59f39b18b65102b28a4f74433a7'),
              'proposed': ('proposed', '123e1fb3ef1b338cada636e71c3e9c7987002402')}
def sha(data):
    return hashlib.sha256(data).hexdigest()
def git(*args):
    return subprocess.check_output(['git', '-C', str(repo), *args])
report = {'source_pins_sha256': sha((root / 'source-pins.json').read_bytes()), 'sources': {}}
for name, commit in commits.items():
    content = git('archive', '--format=tar', commit)
    with tarfile.open(fileobj=io.BytesIO(content)) as archive:
        members = {m.name: m for m in archive.getmembers()}
        def content(name, seen=()):
            assert name not in seen, 'Symlink cycle'
            member = members[name]
            if member.issym():
                target = posixpath.normpath(posixpath.join(posixpath.dirname(name), member.linkname))
                assert not target.startswith(('/', '../'))
                return content(target, (*seen, name))
            assert member.isfile()
            return archive.extractfile(member).read()
        actual = {m.name: sha(content(m.name)) for m in members.values() if m.isfile() or m.issym()}
    assert actual == pins[name]['tracked'], (name, 'tracked content/count mismatch')
    reference_name, reference_commit = references[name]
    native_diff = git('diff', '--name-only', reference_commit, commit, '--', '*.pyx', '*.pxd', '*.pxi',
                      '*.c', '*.cpp', '*.h', '*.cu', 'setup.py', 'pycbc/lib').decode()
    assert not native_diff, native_diff
    build_path = root / (name + '-build.json')
    build = json.loads(build_path.read_text())
    prior_path = previous / (reference_name + '-build.json')
    prior = json.loads(prior_path.read_text())
    assert build['reference_build_sha256'] == sha(prior_path.read_bytes())
    assert prior['commit'] == build['reference_commit'] == reference_commit
    assert prior['native'] == build['native'] == pins[name]['native']
    assert prior['version_sha256'] == build['version_sha256'] == pins[name]['generated_version']
    report['sources'][name] = dict(commit=commit, tracked_files_verified=len(actual),
        every_recorded_tracked_file_matches_git_commit=True,
        all_native_source_and_setup_blobs_match_reference=True,
        reference_commit=reference_commit, reference_build_sha256=sha(prior_path.read_bytes()),
        native_files=len(build['native']), build_sha256=sha(build_path.read_bytes()),
        reused_native_and_generated_version_match_reference_receipt=True)
report.update(status='PASS', scope='Git verifies tracked source bytes and unchanged native source. Native binary and generated-version equality uses prior build receipts, not transferred binary bytes.')
print(json.dumps(report, indent=2))
