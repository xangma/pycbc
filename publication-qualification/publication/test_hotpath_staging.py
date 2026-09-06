"""Reject unbound external dependencies at the optimized archive boundary."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('hotpath_stage', Path(__file__).with_name('stage-evidence.py'))
STAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE)


class InputBindingTests(unittest.TestCase):
    remote = Path('/home/xangma/pycbc-torch-inspiral-reference-20260906')
    frame = '/data/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf'
    digest = 'a' * 64

    def bind(self, inputs, files=None, omitted=None):
        return STAGE.hotpath_inputs(inputs, self.remote, files or {}, self.frame,
                                    {} if omitted is None else omitted)

    def test_exact_external_allowlist(self):
        omitted = {}
        values = dict(STAGE.EXTERNAL_INPUTS, **{self.frame: STAGE.FRAME_SHA256})
        self.assertEqual(self.bind(values, omitted=omitted), set())
        self.assertEqual(omitted, values)

    def test_external_hash_mutation_rejected(self):
        for name in [self.frame, *STAGE.EXTERNAL_INPUTS]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.bind({name: self.digest})

    def test_unlisted_external_rejected(self):
        with self.assertRaises(ValueError):
            self.bind({'/external/arbitrary.so': self.digest})

    def test_portable_bytes_bound(self):
        files = {'report.json': {'sha256': self.digest}}
        self.assertEqual(self.bind({'report.json': self.digest}, files), {'report.json'})
        self.assertEqual(self.bind({str(self.remote / 'report.json'): self.digest}, files), {'report.json'})

    def test_missing_or_changed_portable_bytes_rejected(self):
        for files in ({}, {'report.json': {'sha256': 'b' * 64}}):
            with self.subTest(files=files), self.assertRaises(ValueError):
                self.bind({'report.json': self.digest}, files)

    def test_source_omission_requires_absolute_path(self):
        path = str(self.remote / 'source-v6/pycbc/fft/torchfft.py')
        omitted = {}
        self.assertEqual(self.bind({path: self.digest}, omitted=omitted), set())
        self.assertEqual(omitted, {path: self.digest})
        with self.assertRaises(ValueError):
            self.bind({'source-v6/pycbc/fft/torchfft.py': self.digest})

    def test_conflicting_omission_rejected(self):
        with self.assertRaises(ValueError):
            self.bind({self.frame: STAGE.FRAME_SHA256}, omitted={self.frame: self.digest})

    def test_unsafe_path_and_hash_rejected(self):
        for value in ({}, {'../report.json': self.digest}, {'report.json': 'bad'},
                      {'./report.json': self.digest}, {'report\n.json': self.digest}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.bind(value)


if __name__ == '__main__':
    unittest.main()
