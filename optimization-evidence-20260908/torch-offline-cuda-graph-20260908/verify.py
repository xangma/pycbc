"""Verify or restore publication evidence without executing archived code (stdlib)."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(name):
    require(isinstance(name, str) and name and not name.startswith('/') and
            all(p not in ('', '.', '..') for p in name.split('/')) and
            not any(ord(c) < 32 or ord(c) == 127 or c in '\\:' for c in name),
            'Unsafe path: ' + repr(name))
    return name


def tree(paths):
    for name in paths:
        safe(name)
        require(not any(str(p) in paths for p in PurePosixPath(name).parents),
                'File/directory conflict: ' + name)


def checked(data, record, label):
    require(len(data) == record['bytes'] and sha(data) == record['sha256'],
            'Byte identity differs: ' + label)
    return data


def json_unique(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs)


def unpack(data, expected, label):
    result, seen, dirs = {}, set(), set()
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
        members = archive.getmembers()
        require(len(members) == len(expected), 'Member count differs: ' + label)
        for m, row in zip(members, expected):
            name = m.name[2:] if m.name.startswith('./') else m.name
            if name != '.':
                safe(name)
            require(m.name not in seen and (m.isfile() or m.isdir()),
                    'Duplicate, link or special member: ' + label)
            seen.add(m.name)
            actual = dict(name=m.name, type='file' if m.isfile() else 'directory',
                          bytes=m.size, mode=m.mode, uid=m.uid, gid=m.gid,
                          mtime=m.mtime, pax_headers=m.pax_headers)
            if m.isfile():
                require(name != '.', 'Root file is invalid')
                value = archive.extractfile(m).read()
                actual['sha256'] = sha(value)
                require(name not in result, 'Duplicate normalized member')
                result[name] = value
            else:
                require(name not in dirs, 'Duplicate normalized directory')
                dirs.add(name)
            require(actual == row, 'Archive member differs: ' + label + ':' + m.name)
    require(not set(result) & dirs, 'File/directory alias')
    tree(result)
    return result


def verify(root, destination=None):
    root = Path(root).absolute()
    require(root.is_dir() and not root.is_symlink(), 'Invalid package root')
    actual = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), 'Symlink in package')
        if path.is_dir():
            continue
        require(path.is_file(), 'Nonregular package file')
        name = safe(path.relative_to(root).as_posix())
        if name != 'SHA256SUMS':
            actual[name] = sha(path.read_bytes())
    expected = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        safe(name)
        require(re.fullmatch('[0-9a-f]{64}', digest) and name not in expected and
                name != 'SHA256SUMS', 'Invalid checksum inventory')
        expected[name] = digest
    require(actual == expected, 'Physical package inventory differs')
    manifest = json_unique((root / 'manifest.json').read_bytes())
    require(manifest['schema'] == 'pycbc-publication-evidence-v1', 'Unknown schema')
    files = manifest['files']
    tree(files)
    archive_inventory = json_unique((root / 'archive-inventory.json').read_bytes())
    assets = {}
    for name, record in manifest['transports'].items():
        safe(name)
        packed = root / safe(record['path'])
        require(record['path'] in expected, 'Unsealed transport')
        with gzip.open(packed, 'rb') as stream:
            data = stream.read(record['original']['bytes'] + 1)
        checked(data, record['original'], name)
        assets[('gzip', record['path'])] = data
        members = unpack(data, archive_inventory[name]['members'], name)
        for member, value in members.items():
            assets[('archive', name, member)] = value
    supplement = manifest['supplement']
    with gzip.open(root / safe(supplement['path']), 'rb') as stream:
        data = stream.read(supplement['original']['bytes'] + 1)
    checked(data, supplement['original'], 'supplement')
    for name, value in unpack(data, archive_inventory['supplement']['members'], 'supplement').items():
        assets[('supplement', name)] = value
    restored = {}
    modes = {}
    for name, row in files.items():
        record = row['storage']
        kind = record['kind']
        if kind == 'readable':
            path = safe(record['path'])
            require(path in expected, 'Unsealed readable file')
            value = (root / path).read_bytes()
        elif kind == 'gzip':
            value = assets[(kind, safe(record['path']))]
        elif kind == 'archive':
            value = assets[(kind, safe(record['archive']), safe(record['member']))]
        elif kind == 'supplement':
            value = assets[(kind, safe(record['member']))]
        else:
            raise ValueError('Unknown storage type')
        restored[name] = checked(value, row, name)
        require(type(row['mode']) is int and 0 <= row['mode'] <= 0o777, 'Invalid file mode')
        modes[name] = row['mode']
    sealed_count = 0
    if manifest['original_seal']['present']:
        original = restored['SHA256SUMS']
        require(sha(original) == manifest['original_seal']['sha256'], 'Original seal differs')
        old_seal = {}
        for line in original.decode().splitlines():
            digest, name = line.split('  ', 1)
            safe(name)
            require(name not in old_seal and name in restored, 'Bad original inventory')
            require(sha(restored[name]) == digest, 'Original sealed file differs: ' + name)
            old_seal[name] = digest
        require(set(old_seal) == set(restored) - {'SHA256SUMS'}, 'Original seal coverage differs')
        sealed_count = len(old_seal)
    written = 0
    if destination is not None:
        destination = Path(destination).absolute()
        require(not destination.exists() and not destination.is_symlink(), 'Destination must be unused')
        require(destination.parent.is_dir() and destination.parent.resolve() == destination.parent,
                'Destination parent must exist without symlink aliases')
        require(not destination.is_relative_to(root), 'Destination must be outside package')
        destination.mkdir(mode=0o700)
        for name, value in restored.items():
            target = destination / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(value)
            target.chmod(modes[name])
            checked(target.read_bytes(), files[name], name)
            written += 1
    return dict(state='pass', root=manifest['root_name'], physical_files=len(expected)+1,
                reconstructed_files=len(restored), reconstructed_bytes=sum(len(v) for v in restored.values()),
                original_sealed_entries_verified=sealed_count,
                original_archives=len(manifest['transports']),
                archive_members=sum(len(v['members']) for k,v in archive_inventory.items() if k != 'supplement'),
                written_and_rehashed=written, science_executed=False,
                original_seal_present=manifest['original_seal']['present'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--restore', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.restore), indent=2, sort_keys=True))
