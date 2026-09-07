#!/usr/bin/env python3
"""Verify and reconstruct a gzip-transported evidence supplement using only stdlib."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import stat


PREFIX = 'archive-transport'
MANIFEST = PREFIX + '/manifest.json'
RESTORE = PREFIX + '/restore.py'
README = PREFIX + '/README.md'
THRESHOLD = 90 * 1024 ** 2
MAXIMUM = 100 * 1024 ** 2
FIELDS = ('sha256', 'bytes', 'mode')
HEADER = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_record(path):
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not mode & 0o7000, f'Not a plain file: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(sha256=digest, bytes=path.stat().st_size, mode=stat.S_IMODE(mode))


def relative(name):
    require(isinstance(name, str) and name and not any(c in name for c in '\n\r\\')
            and all(part not in ('', '.', '..') for part in name.split('/'))
            and not Path(name).is_absolute(), f'Unsafe archive path: {name!r}')
    return name


def safe_path(root, name):
    path = root
    for part in relative(name).split('/'):
        path = path / part
        require(not path.is_symlink(), f'Symlink in archive path: {path}')
    return path


def metadata(item):
    return {key: item[key] for key in FIELDS}


def validate_record(item):
    require(set(item) == set(FIELDS) and isinstance(item['sha256'], str)
            and len(item['sha256']) == 64 and all(c in '0123456789abcdef' for c in item['sha256'])
            and type(item['bytes']) is int and item['bytes'] >= 0
            and type(item['mode']) is int and 0 <= item['mode'] <= 0o777,
            'Malformed file metadata')


def verify_gzip(path, original):
    """Bound decompression by the recorded size, checking every original byte."""
    with path.open('rb') as stream:
        require(stream.read(10) == HEADER, f'Noncanonical gzip header: {path}')
    digest, count = hashlib.sha256(), 0
    with gzip.open(path, 'rb') as stream:
        while block := stream.read(1024 * 1024):
            count += len(block)
            require(count <= original['bytes'], f'Expanded file exceeds recorded size: {path}')
            digest.update(block)
    require(count == original['bytes'] and digest.hexdigest() == original['sha256'],
            f'Reconstructed bytes differ: {path}')


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def freeze(path, data, create):
    if path.exists() or path.is_symlink():
        require(file_record(path) == dict(sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data), mode=0o644), f'Frozen transport file changed: {path}')
    else:
        require(create, f'Transport cache is not frozen: {path}')
        with path.open('xb') as stream:
            stream.write(data)
        path.chmod(0o644)


def prepare_transport(logical_files, cache, create=True, threshold=THRESHOLD, maximum=MAXIMUM):
    """Freeze local transport files; return the actual copy inventory and its mapping."""
    require(type(threshold) is int and 0 < threshold <= THRESHOLD
            and type(maximum) is int and threshold < maximum <= MAXIMUM,
            'Invalid archive size policy')
    for name, item in logical_files.items():
        relative(name)
        validate_record(metadata(item))
        require(name != 'SHA256SUMS' and name != PREFIX and not name.startswith(PREFIX + '/'),
                f'Logical evidence collides with transport: {name}')
    oversized = {name: item for name, item in logical_files.items() if item['bytes'] > threshold}
    if not oversized:
        return dict(logical_files), None
    require(not cache.is_symlink(), 'Symlink transport cache')
    if not cache.exists():
        require(create, f'Missing frozen transport cache: {cache}')
        cache.mkdir(parents=True)
    folder = safe_path(cache, PREFIX)
    if not folder.exists():
        require(create, f'Missing frozen transport directory: {folder}')
        folder.mkdir()
    actual = {name: dict(item) for name, item in logical_files.items() if name not in oversized}
    compressed = {}
    for name, item in sorted(oversized.items()):
        original = Path(item['source'])
        require(file_record(original) == metadata(item), f'Original evidence changed: {name}')
        transport_name = PREFIX + '/' + item['sha256'] + '.gz'
        target = safe_path(cache, transport_name)
        if not target.exists():
            require(create, f'Missing frozen compressed input: {name}')
            with original.open('rb') as source, target.open('xb') as raw:
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw,
                                   compresslevel=9, mtime=0) as stream:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)
            target.chmod(0o644)
        packed = file_record(target)
        require(packed['bytes'] < maximum, f'Compressed file still exceeds archive limit: {name}')
        verify_gzip(target, item)
        require(file_record(original) == metadata(item), f'Original changed during compression: {name}')
        require(file_record(target) == packed, f'Transport changed during verification: {name}')
        actual[transport_name] = dict(source=str(target), **packed)
        compressed[name] = dict(path=transport_name, **packed)
    helper_sources = {RESTORE: Path(__file__), README: Path(__file__).with_name('ARCHIVE_TRANSPORT.md')}
    helpers = {}
    for name, source in helper_sources.items():
        target = safe_path(cache, name)
        freeze(target, source.read_bytes(), create)
        helpers[name] = file_record(target)
        actual[name] = dict(source=str(target), **helpers[name])
    manifest = dict(schema='lossless-gzip-evidence-v1', threshold_bytes=threshold,
                    maximum_archive_file_bytes=maximum,
                    gzip=dict(compression_level=9, mtime=0, filename=''),
                    logical_files={name: metadata(item) for name, item in sorted(logical_files.items())},
                    compressed_files=compressed, helpers=helpers)
    path = safe_path(cache, MANIFEST)
    freeze(path, canonical(manifest), create)
    actual[MANIFEST] = dict(source=str(path), **file_record(path))
    require(all(item['bytes'] < maximum for item in actual.values()),
            'An actual archive file exceeds the size limit')
    return dict(sorted(actual.items())), dict(manifest=MANIFEST,
            manifest_sha256=actual[MANIFEST]['sha256'], compressed_files=len(compressed),
            original_bytes=sum(logical_files[name]['bytes'] for name in compressed),
            compressed_bytes=sum(item['bytes'] for name, item in actual.items() if name.endswith('.gz')))


def verify_archive(root):
    """Validate an extracted supplement without modifying it."""
    require(root.is_dir() and not root.is_symlink(), 'Invalid archive root')
    manifest_path = safe_path(root, MANIFEST)
    file_record(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require(manifest['schema'] == 'lossless-gzip-evidence-v1'
            and manifest['gzip'] == dict(compression_level=9, mtime=0, filename='')
            and type(manifest['threshold_bytes']) is int
            and 0 < manifest['threshold_bytes'] <= THRESHOLD
            and type(manifest['maximum_archive_file_bytes']) is int
            and manifest['threshold_bytes'] < manifest['maximum_archive_file_bytes'] <= MAXIMUM,
            'Unrecognized archive transport policy')
    logical, compressed = manifest['logical_files'], manifest['compressed_files']
    require(isinstance(logical, dict) and logical and isinstance(compressed, dict)
            and set(compressed) == {name for name, item in logical.items()
                                   if item['bytes'] > manifest['threshold_bytes']},
            'Incomplete compressed-file mapping')
    require(set(manifest['helpers']) == {RESTORE, README}, 'Incomplete reconstruction helpers')
    actual = dict(manifest['helpers'])
    for name, item in logical.items():
        relative(name)
        validate_record(item)
        require(name != 'SHA256SUMS' and name != PREFIX and not name.startswith(PREFIX + '/'),
                f'Logical path collides with transport: {name}')
        if name in compressed:
            packed = compressed[name]
            require(set(packed) == {'path', *FIELDS}
                    and packed['path'] == PREFIX + '/' + item['sha256'] + '.gz',
                    'Invalid compressed-file path')
            validate_record(metadata(packed))
            require(not safe_path(root, name).exists(), f'Unexpected unpacked original in archive: {name}')
            if packed['path'] in actual:
                require(actual[packed['path']] == metadata(packed), 'Conflicting shared gzip metadata')
            actual[packed['path']] = metadata(packed)
        else:
            actual[name] = item
    for name, expected in actual.items():
        validate_record(expected)
        observed = file_record(safe_path(root, name))
        # Git/extraction may normalize modes; original modes are restored from the manifest.
        require(expected['bytes'] < manifest['maximum_archive_file_bytes']
                and all(observed[key] == expected[key] for key in ('sha256', 'bytes')),
                f'Archive file changed: {name}')
    for name, packed in compressed.items():
        verify_gzip(safe_path(root, packed['path']), logical[name])
    require(canonical(json.loads(manifest_path.read_text())) == canonical(manifest),
            'Transport manifest changed during verification')
    return manifest


def restore_archive(root, destination):
    require(not destination.exists() and not destination.is_symlink(),
            'Reconstruction destination must be a new directory')
    require(not destination.resolve().is_relative_to(root.resolve()),
            'Reconstruction must be outside the immutable archive')
    manifest = verify_archive(root)
    destination.mkdir(parents=True)
    for name, item in manifest['logical_files'].items():
        target = safe_path(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        packed = manifest['compressed_files'].get(name)
        source = safe_path(root, packed['path'] if packed else name)
        opener = gzip.open if packed else open
        with opener(source, 'rb') as stream, target.open('xb') as output:
            shutil.copyfileobj(stream, output, length=1024 * 1024)
        target.chmod(item['mode'])
        require(file_record(target) == item, f'Restored file differs: {name}')
    verify_archive(root)
    return len(manifest['logical_files'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify_archive(args.root)
        print(json.dumps(dict(status='verified', logical_files=len(manifest['logical_files']),
                              compressed_files=len(manifest['compressed_files']))))
    else:
        count = restore_archive(args.root, args.destination)
        print(json.dumps(dict(status='restored', destination=str(args.destination), logical_files=count)))


if __name__ == '__main__':
    main()
