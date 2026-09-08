"""Check acquired tracked-source hashes against the two real Git commits."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', required=True)
    parser.add_argument('--source-pins', default='acquisition/source-pins.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    pins_path = Path(args.source_pins)
    pins = json.loads(pins_path.read_text())
    expected = {'original': '40e94792b3edf59f39b18b65102b28a4f74433a7',
                'proposed': '123e1fb3ef1b338cada636e71c3e9c7987002402'}
    assert set(pins) == set(expected)
    result = {'source_pins_sha256': sha(pins_path.read_bytes()), 'sources': {}}
    for name, commit in expected.items():
        assert pins[name]['info']['commit'] == commit
        archive = subprocess.check_output(['git', '-C', args.repository, 'archive', commit])
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            for path, digest in pins[name]['tracked'].items():
                file = source.extractfile(source.getmember(path))
                assert file is not None and sha(file.read()) == digest, (name, path)
        result['sources'][name] = {
            'commit': commit, 'tracked_files': len(pins[name]['tracked']),
            'tracked_manifest_sha256': sha(json.dumps(pins[name]['tracked'], sort_keys=True).encode()),
            'git_archive_sha256': sha(archive), 'all_recorded_tracked_bytes_match_commit': True}
    with Path(args.output).open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print('PASS: both acquired tracked-source manifests match their actual Git commits')


if __name__ == '__main__':
    main()
