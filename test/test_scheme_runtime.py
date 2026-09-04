import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import pycbc.scheme as scheme


class _OpenMPRuntime:
    def __init__(self, max_threads=1):
        self.max_threads = max_threads
        self.thread_counts = []

    def omp_get_max_threads(self):
        return self.max_threads

    def omp_set_num_threads(self, count):
        self.max_threads = count
        self.thread_counts.append(count)


@pytest.fixture(autouse=True)
def _clear_libgomp_cache():
    scheme._resolve_libgomp.cache_clear()
    yield
    scheme._resolve_libgomp.cache_clear()


@pytest.mark.parametrize("explicit", (None, "INTEL", "SEQUENTIAL", "GNU"))
def test_pycbc_defaults_mkl_threading_without_overriding_user(explicit):
    env = os.environ.copy()
    env.pop("MKL_THREADING_LAYER", None)
    if explicit is not None:
        env["MKL_THREADING_LAYER"] = explicit
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import pycbc; "
                f"assert os.environ['MKL_THREADING_LAYER'] == {explicit or 'GNU'!r}"
            ),
        ],
        check=True,
        env=env,
    )


@pytest.mark.parametrize("explicit", (None, "INTEL", "SEQUENTIAL", "GNU"))
def test_cpu_scheme_preserves_mkl_threading_configuration(monkeypatch, explicit):
    monkeypatch.delenv("MKL_THREADING_LAYER", raising=False)
    if explicit is not None:
        monkeypatch.setenv("MKL_THREADING_LAYER", explicit)
    monkeypatch.setattr(scheme.pycbc, "HAVE_MKL", True)
    monkeypatch.setattr(scheme, "_resolve_libgomp", lambda: _OpenMPRuntime())
    with scheme.CPUScheme():
        assert os.environ["MKL_THREADING_LAYER"] == (explicit or "GNU")
    assert os.environ["MKL_THREADING_LAYER"] == (explicit or "GNU")


def test_torch_scheme_num_threads_validation():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="num_threads must be positive"):
        scheme.TorchScheme("cpu", num_threads=0)
    with pytest.raises(ValueError, match="num_threads must be positive"):
        scheme.TorchScheme("cpu", num_threads=-1)


def test_torch_scheme_cpu_threads_restore_after_nested_rejection_and_error(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    runtime = _OpenMPRuntime(max_threads=6)
    monkeypatch.setattr(
        scheme,
        "get_ctypes_library",
        lambda name, packages, mode=None: runtime,
    )

    orig_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        with scheme.TorchScheme("cpu"):
            assert torch.get_num_threads() == 8
            assert runtime.max_threads == 6
        assert torch.get_num_threads() == 8
        assert runtime.max_threads == 6

        context = scheme.TorchScheme("cpu", num_threads=2)
        with pytest.raises(ValueError, match="body failure"):
            with context:
                assert scheme.mgr.state is context
                assert torch.get_num_threads() == 2
                assert runtime.max_threads == 2

                nested = scheme.TorchScheme("cpu", num_threads=4)
                with pytest.raises(RuntimeError, match="state is locked"):
                    with nested:
                        pass

                assert scheme.mgr.state is context
                assert torch.get_num_threads() == 2
                assert runtime.max_threads == 2
                raise ValueError("body failure")

        assert torch.get_num_threads() == 8
        assert runtime.max_threads == 6
        assert runtime.thread_counts == [2, 6]
    finally:
        torch.set_num_threads(orig_threads)


def test_torch_scheme_parsing():
    assert scheme._parse_torch_scheme_extra(None) == (None, None)
    assert scheme._parse_torch_scheme_extra("cpu") == ("cpu", None)
    assert scheme._parse_torch_scheme_extra("cuda:0") == ("cuda:0", None)
    assert scheme._parse_torch_scheme_extra("4") == ("cpu", 4)
    assert scheme._parse_torch_scheme_extra("cpu:2") == ("cpu", 2)


@pytest.mark.parametrize(
    ("scheme_name", "device_id", "expected_device", "expected_threads"),
    (
        ("torch", 7, None, None),
        ("torch:cpu", 7, "cpu", None),
        ("torch:3", 7, "cpu", 3),
        ("torch:cpu:2", 7, "cpu", 2),
        ("torch:cuda", 7, "cuda:7", None),
        ("torch:cuda:4", 7, "cuda:4", None),
        ("torch:mps", 0, "mps:0", None),
        ("torch:mps:0", 7, "mps:0", None),
    ),
)
def test_torch_cli_device_selection(
    monkeypatch,
    scheme_name,
    device_id,
    expected_device,
    expected_threads,
):
    class FakeTorchScheme:
        def __init__(self, device=None, num_threads=None):
            self.device = device
            self.num_threads = num_threads
            self.torch_device = device

    monkeypatch.setattr(scheme, "TorchScheme", FakeTorchScheme)
    context = scheme.from_cli(
        SimpleNamespace(
            processing_scheme=scheme_name,
            processing_device_id=device_id,
        )
    )

    assert context.device == expected_device
    assert context.num_threads == expected_threads


def test_cuda_cli_device_selection_is_unchanged(monkeypatch):
    class FakeCUDAScheme:
        def __init__(self, device_num):
            self.device_num = device_num

    monkeypatch.setattr(scheme, "CUDAScheme", FakeCUDAScheme)
    context = scheme.from_cli(
        SimpleNamespace(processing_scheme="cuda", processing_device_id=7)
    )

    assert context.device_num == 7
