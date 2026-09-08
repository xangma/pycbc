"""Independently verify the final local tar, manifest, and extracted bytes."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile


EXPECTED_ARCHIVE = '3c7cff78082c7b344a38d13d216ae1c647315720eea1b4690b4e969d618b4af6'
EXPECTED_MANIFEST = '25c482d8fff30a1b0d94206bd1032b0a27479f645f789b32d2e8df1a6337b2b2'
HERE = Path(__file__).resolve().parent


def sha_stream(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def sha(path):
    with path.open('rb') as stream:
        return sha_stream(stream)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, ('Duplicate JSON key', key)
        result[key] = value
    return result


root = Path(sys.argv[1]).resolve()
archive = root.parent / 'final-evidence.tar.gz'
download = root.parent / 'download-verification.json'
manifest_path = root / 'final-evidence-sha256.json'
manifest = json.loads(manifest_path.read_text(), object_pairs_hook=unique_pairs)
assert len(manifest) == 292
assert sha(archive) == EXPECTED_ARCHIVE
assert sha(manifest_path) == EXPECTED_MANIFEST
assert json.loads(download.read_text(), object_pairs_hook=unique_pairs) == dict(
    status='PASS', files=292, archive_sha256=EXPECTED_ARCHIVE,
    manifest_sha256=EXPECTED_MANIFEST)
expected = dict(manifest, **{'final-evidence-sha256.json': EXPECTED_MANIFEST})
for name, digest in expected.items():
    path = PurePosixPath(name)
    assert not path.is_absolute() and '..' not in path.parts and str(path) == name
    file = root / name
    assert file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(root)
    assert sha(file) == digest, ('Extracted file differs', name)
files = {str(file.relative_to(root)) for file in root.rglob('*') if file.is_file()}
assert files == set(expected), ('Unexpected/missing extracted files', files ^ set(expected))
tar_hashes = {}
with tarfile.open(archive, 'r:gz') as tar:
    for member in tar:
        assert member.name in expected and member.isfile() and member.name not in tar_hashes
        with tar.extractfile(member) as stream:
            tar_hashes[member.name] = sha_stream(stream)
assert tar_hashes == expected, 'Tar members differ from manifest/extracted bytes'
verification_path = Path(sys.argv[2]).resolve()
assert verification_path.is_relative_to(HERE)
verification = json.loads(verification_path.read_text(), object_pairs_hook=unique_pairs)
assert verification['evidence_root'] == str(root)
assert verification['verifier_sha256'] == sha(HERE / 'verify-results.py')
for name, digest in verification['input_evidence_sha256'].items():
    assert expected[name] == digest, ('Verifier evidence differs from export manifest', name)
primary_source = root.parent / 'source-review.json'
assert sha(primary_source) == sha(HERE / 'source-review.json')
assert sha(primary_source) == verification['independent_source_review_receipt']['sha256']
print(json.dumps(dict(
    status='PASS', archive_sha256=EXPECTED_ARCHIVE, manifest_sha256=EXPECTED_MANIFEST,
    download_verification_sha256=sha(download), source_review_sha256=sha(primary_source),
    scientific_verification_file=verification_path.name,
    scientific_verification_sha256=sha(verification_path),
    verifier_sha256=sha(HERE / 'verify-results.py'),
    transfer_verifier_sha256=sha(Path(__file__).resolve()),
    manifest_files=292, tar_members=293, extracted_files=293,
    consumed_evidence_files=len(verification['input_evidence_sha256']),
    unconsumed_manifest_files=sorted(set(manifest) - set(verification['input_evidence_sha256'])),
    scope='Local tar and all extracted bytes independently match the supplied archive SHA and '
          'export manifest. Remote provenance is the primary download receipt; no remote read '
          'or mutation was performed. Hash-only auxiliary files are not benchmark measurements.'
), indent=2))
