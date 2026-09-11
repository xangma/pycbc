import copy
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
def test_pycbc_preserves_mkl_threading_configuration(explicit):
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
                "assert os.environ.get('MKL_THREADING_LAYER') == "
                f"{explicit!r}"
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
    def no_cpu_runtime_load():
        pytest.fail("CPU entry must not adopt Torch OpenMP setup")

    monkeypatch.setattr(scheme, "_resolve_libgomp", no_cpu_runtime_load)
    with scheme.CPUScheme():
        assert os.environ.get("MKL_THREADING_LAYER") == explicit
    assert os.environ.get("MKL_THREADING_LAYER") == explicit


def test_torch_scheme_num_threads_validation():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="num_threads must be positive"):
        scheme.TorchScheme("cpu", num_threads=0)
    with pytest.raises(ValueError, match="num_threads must be positive"):
        scheme.TorchScheme("cpu", num_threads=-1)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_torch_scheme_preserves_context_when_deepcopying_arrays(
    monkeypatch,
    device,
):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA is unavailable")
    from pycbc.types import Array, FrequencySeries, TimeSeries

    runtime = _OpenMPRuntime(max_threads=6)
    monkeypatch.setattr(
        scheme,
        "get_ctypes_library",
        lambda name, packages, mode=None: runtime,
    )
    orig_threads = torch.get_num_threads()
    context = scheme.TorchScheme(device, num_threads=2)
    try:
        torch.set_num_threads(3)
        with context:
            out_vals = {
                "array": Array([1.0, 2.0, 3.0], dtype="float32"),
                "frequency": FrequencySeries(
                    [1.0 + 2.0j, 3.0 + 4.0j],
                    delta_f=0.25,
                    epoch=1234567890.125,
                    dtype="complex64",
                ),
                "time": TimeSeries(
                    [4.0, 5.0, 6.0],
                    delta_t=0.125,
                    epoch=1234567890.25,
                    dtype="float64",
                ),
            }
            for array in out_vals.values():
                array.metadata = {"labels": ["original"]}
            out_vals["alias"] = out_vals["array"]
            thread_state = context._thread_state

            copied = copy.deepcopy(out_vals)

            assert copied is not out_vals
            assert copied["alias"] is copied["array"]
            assert context._thread_state is thread_state
            assert scheme.mgr.state is context
            assert torch.get_num_threads() == (2 if device == "cpu" else 3)
            assert runtime.max_threads == (2 if device == "cpu" else 6)
            for name in ("array", "frequency", "time"):
                original, duplicate = out_vals[name], copied[name]
                assert duplicate is not original
                assert type(duplicate) is type(original)
                assert duplicate._scheme is original._scheme is context
                assert duplicate.data.tensor.device == original.data.tensor.device
                assert (
                    duplicate.data.tensor.data_ptr() != original.data.tensor.data_ptr()
                )
                expected = original.data.tensor.clone()
                torch.testing.assert_close(duplicate.data.tensor, expected)
                duplicate[0] += 10
                torch.testing.assert_close(original.data.tensor, expected)
                duplicate.metadata["labels"].append("copied")
                assert original.metadata == {"labels": ["original"]}

            assert copied["frequency"].delta_f == out_vals["frequency"].delta_f
            assert copied["time"].delta_t == out_vals["time"].delta_t
            for name in ("frequency", "time"):
                assert copied[name].start_time == out_vals[name].start_time
                copied[name].start_time += 1
                assert copied[name].start_time == out_vals[name].start_time + 1

        assert scheme.mgr.state is scheme.default_context
        assert context._thread_state is None
        assert torch.get_num_threads() == 3
        assert runtime.max_threads == 6
        assert runtime.thread_counts == ([2, 6] if device == "cpu" else [])
        inactive_copy = copy.deepcopy(out_vals)
        assert inactive_copy["array"]._scheme is context
        assert scheme.mgr.state is scheme.default_context
        assert context._thread_state is None
        assert torch.get_num_threads() == 3
        assert runtime.max_threads == 6
    finally:
        torch.set_num_threads(orig_threads)


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
