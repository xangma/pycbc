"""Small failure controls for the publication integrity and extraction gates."""
import hashlib
from pathlib import Path
import tarfile
import tempfile
import types
import unittest

verifier = types.ModuleType('verify_bundle')
source = Path(__file__).with_name('verify_bundle.py')
exec(compile(source.read_bytes(), str(source), 'exec'), verifier.__dict__)


class BundleControls(unittest.TestCase):
    def test_canonical_paths(self):
        self.assertEqual(str(verifier.safe_path('root/file')), 'root/file')
        for name in ('', '/root/file', 'root/../file', 'root//file', './root', 'root\\file'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unsafe path'):
                verifier.safe_path(name)

    def test_missing_corrupt_and_unlisted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sha = hashlib.sha256(b'original').hexdigest()
            (root / 'SHA256SUMS').write_text(sha + '  part\n')
            with self.assertRaisesRegex(ValueError, 'inventory differs'):
                verifier.check_publication(root)
            (root / 'part').write_bytes(b'original')
            self.assertEqual(verifier.check_publication(root), {'part': sha})
            (root / 'part').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                verifier.check_publication(root)
            (root / 'part').write_bytes(b'original')
            (root / 'nested').mkdir()
            (root / 'nested/SHA256SUMS').write_text('unexpected')
            with self.assertRaisesRegex(ValueError, 'inventory differs'):
                verifier.check_publication(root)

    def members(self, *entries):
        inventory = dict(root=verifier.EXPECTED_ROOT, member_count=len(entries),
                         members=[verifier.member_row(m) for m in entries])
        return verifier.validate_members(list(entries), inventory)

    def member(self, suffix, kind=tarfile.REGTYPE, target=''):
        m = tarfile.TarInfo(verifier.EXPECTED_ROOT + '/' + suffix)
        m.type, m.linkname = kind, target
        return m

    def test_internal_relative_symlink_allowed(self):
        target = self.member('file')
        link = self.member('sub/link', tarfile.SYMTYPE, '../file')
        _, links = self.members(target, link)
        self.assertEqual(links, {link.name: '../file'})

    def test_escaping_symlink_rejected(self):
        for target in ('../../outside', '/outside'):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.members(self.member('link', tarfile.SYMTYPE, target))

    def test_child_below_symlink_rejected(self):
        with self.assertRaisesRegex(ValueError, 'below a symlink'):
            self.members(self.member('file'), self.member('link', tarfile.SYMTYPE, 'file'),
                         self.member('link/child'))

    def test_duplicate_members_rejected(self):
        m = self.member('file')
        inventory = dict(root=verifier.EXPECTED_ROOT, member_count=1,
                         members=[verifier.member_row(m)])
        with self.assertRaisesRegex(ValueError, 'Duplicate archive member'):
            verifier.validate_members([m, m], inventory)

    def test_hardlinks_and_specials_rejected(self):
        for kind in (tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'special files rejected'):
                self.members(self.member('special', kind, 'file'))


if __name__ == '__main__':
    unittest.main()
