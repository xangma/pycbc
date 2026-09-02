import ctypes
import os
import subprocess
import sys

import pytest

import pycbc.scheme as scheme


class _OpenMPRuntime:
    def __init__(self):
        self.thread_counts = []

    def omp_set_num_threads(self, count):
        self.thread_counts.append(count)


@pytest.fixture(autouse=True)
def _clear_libgomp_cache():
    scheme._resolve_libgomp.cache_clear()
    yield
    scheme._resolve_libgomp.cache_clear()


def test_pycbc_selects_gnu_mkl_threading_before_optional_backends():
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "INTEL"
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import pycbc; "
                "assert os.environ['MKL_THREADING_LAYER'] == 'GNU'"
            ),
        ],
        check=True,
        env=env,
    )


def test_cpu_scheme_configures_openmp_runtime(monkeypatch):
    runtime = _OpenMPRuntime()
    load_calls = []

    def load_runtime(name, packages, mode=None):
        assert scheme.os.environ["MKL_THREADING_LAYER"] == "GNU"
        load_calls.append((name, packages, mode))
        return runtime

    monkeypatch.setattr(scheme, "get_ctypes_library", load_runtime)
    monkeypatch.setattr(scheme.pycbc, "HAVE_MKL", True)
    monkeypatch.setenv("MKL_THREADING_LAYER", "INTEL")
    monkeypatch.setenv("OMP_NUM_THREADS", "7")

    with scheme.CPUScheme(num_threads=3):
        assert scheme.mgr.state.num_threads == 3
        assert scheme.current_prefix() == "cpu"
        assert scheme.os.environ["MKL_THREADING_LAYER"] == "GNU"
        assert scheme.os.environ["OMP_NUM_THREADS"] == "3"

    with scheme.CPUScheme(num_threads=5):
        assert scheme.mgr.state.num_threads == 5
        assert scheme.os.environ["MKL_THREADING_LAYER"] == "GNU"
        assert scheme.os.environ["OMP_NUM_THREADS"] == "5"

    assert load_calls == [("gomp", ["gomp"], ctypes.RTLD_GLOBAL)]
    assert runtime.thread_counts == [3, 1, 5, 1]
    assert scheme.os.environ["OMP_NUM_THREADS"] == "1"


def test_cpu_scheme_retries_failed_openmp_runtime_discovery(monkeypatch):
    load_calls = []

    def fail_to_load(name, packages, mode=None):
        load_calls.append((name, packages, mode))
        raise OSError("libgomp unavailable")

    monkeypatch.setattr(scheme, "get_ctypes_library", fail_to_load)
    monkeypatch.setenv("OMP_NUM_THREADS", "7")

    for num_threads in (3, 5):
        context = scheme.CPUScheme(num_threads=num_threads)
        with context:
            assert context._libgomp is None
            assert scheme.os.environ["OMP_NUM_THREADS"] == str(num_threads)
        assert scheme.os.environ["OMP_NUM_THREADS"] == "1"

    expected_call = ("gomp", ["gomp"], ctypes.RTLD_GLOBAL)
    assert load_calls == [expected_call, expected_call]


def test_cpu_scheme_caches_missing_openmp_runtime(monkeypatch):
    load_calls = []

    def missing_runtime(name, packages, mode=None):
        load_calls.append((name, packages, mode))
        return None

    monkeypatch.setattr(scheme, "get_ctypes_library", missing_runtime)

    for num_threads in (3, 5):
        context = scheme.CPUScheme(num_threads=num_threads)
        with context:
            assert context._libgomp is None
            assert scheme.os.environ["OMP_NUM_THREADS"] == str(num_threads)
        assert scheme.os.environ["OMP_NUM_THREADS"] == "1"

    assert load_calls == [("gomp", ["gomp"], ctypes.RTLD_GLOBAL)]


def test_cpuonly_reports_decorated_function(monkeypatch):
    @scheme.cpuonly
    def cpu_function():
        return "cpu"

    assert cpu_function() == "cpu"

    monkeypatch.setattr(scheme.mgr, "state", object())
    with pytest.raises(
        TypeError,
        match="cpu_function can only be called from a CPU processing scheme",
    ):
        cpu_function()


def test_torch_scheme_num_threads_validation():
    with pytest.raises(ValueError, match="num_threads must be positive"):
        scheme.TorchScheme("cpu", num_threads=0)
    with pytest.raises(ValueError, match="num_threads must be positive"):
        scheme.TorchScheme("cpu", num_threads=-1)


def test_torch_scheme_cpu_threads_clamp_and_restore():
    import torch

    orig_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        # Without num_threads configured, enter/exit does not modify threads
        with scheme.TorchScheme("cpu"):
            assert torch.get_num_threads() == 8
        assert torch.get_num_threads() == 8

        # With num_threads configured, enter sets num_threads and exit restores
        with scheme.TorchScheme("cpu", num_threads=2):
            assert scheme.mgr.state.num_threads == 2
            assert torch.get_num_threads() == 2
        assert torch.get_num_threads() == 8
    finally:
        torch.set_num_threads(orig_threads)


def test_torch_scheme_parsing():
    assert scheme._parse_torch_scheme_extra(None) == (None, None)
    assert scheme._parse_torch_scheme_extra("cpu") == ("cpu", None)
    assert scheme._parse_torch_scheme_extra("cuda:0") == ("cuda:0", None)
    assert scheme._parse_torch_scheme_extra("4") == ("cpu", 4)
    assert scheme._parse_torch_scheme_extra("cpu:2") == ("cpu", 2)


def test_waveform_cpu_single_waveform_thread_clamping():
    import torch
    from pycbc.waveform.torch_waveform_registry import (
        try_torch_native_waveform,
    )

    orig_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(8)

        # 1. Under TorchScheme without explicit num_threads ->
        # clamped to 1 during waveform gen and restored to 8
        with scheme.TorchScheme("cpu"):
            assert scheme.mgr.state.num_threads is None
            params = {
                "approximant": "TaylorF2",
                "mass1": 1.4,
                "mass2": 1.4,
                "f_lower": 30.0,
                "delta_f": 1.0,
            }
            out = try_torch_native_waveform("fd", params)
            assert out is not None
            assert torch.get_num_threads() == 8

        # 2. Under TorchScheme with explicit num_threads=4 ->
        # keeps 4 during generation
        with scheme.TorchScheme("cpu", num_threads=4):
            assert scheme.mgr.state.num_threads == 4
            assert torch.get_num_threads() == 4
            params = {
                "approximant": "TaylorF2",
                "mass1": 1.4,
                "mass2": 1.4,
                "f_lower": 30.0,
                "delta_f": 1.0,
            }
            out = try_torch_native_waveform("fd", params)
            assert out is not None
            assert torch.get_num_threads() == 4

        assert torch.get_num_threads() == 8
    finally:
        torch.set_num_threads(orig_threads)
