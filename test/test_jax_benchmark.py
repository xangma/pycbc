"""Opt-in host profiling remains independent of accelerator libraries."""

import subprocess
import sys
from pathlib import Path

import pytest

from pycbc.benchmark import host_stage


def test_host_stage_balances_failure_and_keeps_profiling_optional(monkeypatch):
    records = []

    def emit(stage, event, **metadata):
        records.append((stage, event, metadata))

    monkeypatch.setattr('pycbc.benchmark.stage_event', emit)
    with host_stage('coincidence', sequence=3):
        pass
    assert records == []
    monkeypatch.setenv('PYCBC_BENCHMARK_STAGES', '1')
    with pytest.raises(ValueError, match='science failed'):
        with host_stage('coincidence', sequence=3):
            raise ValueError('science failed')
    assert records == [
        ('coincidence', 'start', {'synchronize': False, 'sequence': 3}),
        ('coincidence', 'end', {'synchronize': False, 'sequence': 3})]


def test_backend_and_disabled_host_profiler_do_not_import_jax():
    # The byte transport needs MPI, but no JAX import is necessary to load its
    # module or use the disabled host-only profiler in CPU tooling.
    code = '''
import importlib.abc
import importlib.util
import os
import sys
class NoJAX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jax" or fullname.startswith("jax."):
            raise RuntimeError("Unexpected JAX dependency")
sys.meta_path.insert(0, NoJAX())
os.environ.pop("PYCBC_BENCHMARK_STAGES", None)
spec = importlib.util.spec_from_file_location("transport", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
with module.host_stage("coincidence"):
    pass
assert "jax" not in sys.modules
assert "mpi4py" not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(Path(__file__).parents[1] / 'pycbc/benchmark.py')],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
