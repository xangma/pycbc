"""Verify the acquired tracked bytes against the two measured Git commits."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import posixpath
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', required=True)
    parser.add_argument('--source-pins', default='acquisition/source-pins.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    pins_path = Path(args.source_pins)
    pins = json.loads(pins_path.read_text())
    expected = {'corrected': '66789ac4a7468094b0cc3ca1498a1de67e0311f6',
                'proposed': 'f582b6fd250d0b82612492979e01e645d5c07afc'}
    assert set(pins) == set(expected)
    result = dict(status='PASS', source_pins_sha256=hashlib.sha256(pins_path.read_bytes()).hexdigest(), sources={})
    for name, commit in expected.items():
        assert pins[name]['info'] == dict(commit=commit, status='', tracked_diff='')
        archive = subprocess.check_output(['git', '-C', args.repository, 'archive', commit])
        links = {}
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            members = {m.name: m for m in source if m.isfile() or m.issym()}
            assert set(members) == set(pins[name]['tracked']), name
            for path, digest in pins[name]['tracked'].items():
                resolved = path
                seen = set()
                while members[resolved].issym():
                    assert resolved not in seen
                    seen.add(resolved)
                    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(resolved), members[resolved].linkname))
                    assert resolved in members
                if resolved != path:
                    links[path] = resolved
                data = source.extractfile(members[resolved]).read()
                assert hashlib.sha256(data).hexdigest() == digest, (name, path)
        result['sources'][name] = dict(commit=commit, tracked_count=len(members),
            symlink_targets=links, all_recorded_tracked_bytes_match_commit=True,
            git_archive_sha256=hashlib.sha256(archive).hexdigest())
    result['scope'] = 'Git-tracked source only. Native binaries and generated version files are separately pinned by acquisition records.'
    with Path(args.output).open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print('PASS: all acquired tracked source bytes match the measured Git commits')


if __name__ == '__main__':
    main()
