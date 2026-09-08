"""Run original CPU regression tests with Torch discovery/import blocked."""
import importlib.abc
import sys


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError('Torch deliberately unavailable for CPU isolation check')
        return None


sys.meta_path.insert(0, NoTorch())
import pytest

result = pytest.main(['-q', '-ra', *sys.argv[1:]])
assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)
print('Confirmed: no Torch module imported.')
raise SystemExit(result)
