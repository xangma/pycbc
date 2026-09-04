# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Focused parity and allocation-path tests for Torch search kernels."""

from types import SimpleNamespace

import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.events import coinc
from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.types.array_torch import TorchArrayData


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


@pytest.mark.parametrize("method", ("python", "cython"))
def test_cluster_over_time_stays_on_torch_device(monkeypatch, method):
    host_stat = np.array(
        [5.0, 2.0, 7.0, 7.0, -1.0, 4.0, 8.0, 3.0]
    )
    host_time = np.array(
        [4.0, 0.0, 1.0, 1.25, 8.0, 4.25, 7.75, 12.0]
    )
    expected = coinc.cluster_over_time(
        host_stat, host_time, window=0.5, method=method
    )

    def reject_host_path(*_args, **_kwargs):
        raise AssertionError("clustering used the NumPy/Cython path")

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("clustering copied or synchronized Torch data")

    with scheme.TorchScheme("cpu"):
        torch_stat = Array(host_stat)
        torch_time = Array(host_time)
        monkeypatch.setattr(coinc, "timecluster_cython", reject_host_path)
        monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
        monkeypatch.setattr(torch.Tensor, "cpu", reject_host_transfer)
        monkeypatch.setattr(torch.Tensor, "item", reject_host_transfer)
        actual = coinc.cluster_over_time(
            torch_stat, torch_time, window=0.5, method=method
        )

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == "cpu"
    assert actual._data.tensor.dtype == torch.int64
    np.testing.assert_array_equal(
        actual._data.tensor.detach().numpy(), expected
    )


def test_cluster_over_time_raw_torch_nan_and_validation():
    times = torch.tensor([0.0, 0.1, 0.2, 2.0], dtype=torch.float64)
    cases = (
        ([1.0, np.nan, 3.0, 4.0], [1, 3], [2, 3]),
        ([np.nan, 4.0, 3.0, 2.0], [0, 3], [0, 3]),
        ([1.0, 2.0, np.nan, np.nan], [2, 3], [1, 3]),
    )
    for statistics, python_expected, cython_expected in cases:
        stat = torch.tensor(statistics, dtype=torch.float64)
        for method, expected in (
            ("python", python_expected),
            ("cython", cython_expected),
        ):
            actual = coinc.cluster_over_time(
                stat, times, window=0.5, method=method
            )
            assert isinstance(actual, torch.Tensor)
            np.testing.assert_array_equal(actual.numpy(), expected)

    empty = coinc.cluster_over_time(
        torch.empty(0, dtype=torch.float64),
        torch.empty(0, dtype=torch.float64),
        window=0.5,
    )
    assert isinstance(empty, torch.Tensor)
    assert empty.dtype == torch.int64
    assert empty.numel() == 0

    with pytest.raises(NotImplementedError):
        coinc.cluster_over_time(
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            window=0.5,
            argmax=lambda value: value.argmax(),
        )


def test_coincidence_torch_backend_public_dispatch():
    """Torch public APIs retain parity after moving to their backend module."""
    time1 = np.array([0.0, 0.6, 1.2, 2.5], dtype=np.float64)
    time2 = np.array([0.1, 0.8, 1.1, 3.0], dtype=np.float64)
    expected = coinc.time_coincidence(time1, time2, 0.25)
    actual = coinc.time_coincidence(
        torch.as_tensor(time1), torch.as_tensor(time2), 0.25
    )
    for torch_value, numpy_value in zip(actual, expected):
        assert isinstance(torch_value, torch.Tensor)
        np.testing.assert_array_equal(torch_value.numpy(), numpy_value)

    stat = np.array([2.0, 7.0, 4.0, 8.0], dtype=np.float64)
    slide_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    expected = coinc.cluster_coincs(
        stat, time1, time2, slide_ids, 1.0, 0.5
    )
    actual = coinc.cluster_coincs(
        torch.as_tensor(stat),
        torch.as_tensor(time1),
        torch.as_tensor(time2),
        torch.as_tensor(slide_ids),
        1.0,
        0.5,
    )
    assert isinstance(actual, torch.Tensor)
    np.testing.assert_array_equal(actual.numpy(), expected)

    detector_times = (
        time1 + 10.0,
        time2 + 10.0,
        np.array([-1.0, 10.7, 11.0, 12.9], dtype=np.float64),
    )
    expected = coinc.cluster_coincs_multiifo(
        stat, detector_times, slide_ids, 1.0, 0.5
    )
    actual = coinc.cluster_coincs_multiifo(
        torch.as_tensor(stat),
        tuple(torch.as_tensor(value) for value in detector_times),
        torch.as_tensor(slide_ids),
        1.0,
        0.5,
    )
    assert isinstance(actual, torch.Tensor)
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_cuda_graph_capture_delegates_to_torch_backend(monkeypatch):
    """The generic control exposes only the backend delegation point."""
    from pycbc.filter import matchedfilter_torch
    from pycbc.filter.matchedfilter import MatchedFilterControl

    marker = object()
    calls = []

    def capture(control, segnum, window, template_norm):
        calls.append((control, segnum, window, template_norm))
        return marker

    monkeypatch.setattr(
        matchedfilter_torch, "capture_symmetric_cuda_graph", capture
    )
    control = object.__new__(MatchedFilterControl)
    control.threshold_and_clusterers = [
        SimpleNamespace(series=SimpleNamespace(is_cuda=True))
    ]

    result = control.capture_cuda_graph_symm(0, 64, 2.0)

    assert result is marker
    assert calls == [(control, 0, 64, 2.0)]


def test_correlators_write_multiplication_directly_to_output(monkeypatch):
    from pycbc.filter import matchedfilter_torch

    x_values = np.array([1 + 2j, -3 + 0.5j, 0.25 - 4j], np.complex64)
    y_values = np.array([2 - 1j, 0.75 + 5j, -2 - 3j], np.complex64)
    expected = np.conj(x_values) * y_values

    with scheme.TorchScheme("cpu"):
        x = Array(x_values)
        y = Array(y_values)
        outputs = [Array(np.empty_like(x_values)) for _ in range(3)]

        original_mul = torch.mul
        direct_outputs = []

        def record_mul(left, right, *, out=None):
            direct_outputs.append(out)
            return original_mul(left, right, out=out)

        monkeypatch.setattr(matchedfilter_torch.torch, "mul", record_mul)
        matchedfilter_torch.correlate(x, y, outputs[0])
        matchedfilter_torch.TorchCorrelator(x, y, outputs[1]).correlate()
        batch = SimpleNamespace(xs=[x], zs=[outputs[2]])
        matchedfilter_torch.batch_correlate_execute(batch, y)

        assert len(direct_outputs) == len(outputs)
        assert all(a.data_ptr() == b.data_ptr() for a, b in zip(direct_outputs, [item._data.tensor for item in outputs]))
        for output in outputs:
            np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize(
    ("operation", "itype", "otype", "inverse"),
    [
        ("fft", "complex", "complex", False),
        ("rfft", "real", "complex", False),
        ("ifft", "complex", "complex", True),
        ("irfft", "complex", "real", True),
    ],
)
def test_fft_writes_out_of_place_results_directly(
    monkeypatch, operation, itype, otype, inverse
):
    from pycbc.fft import torchfft

    rng = np.random.default_rng(1701)
    size = 32
    if operation == "rfft":
        input_values = rng.normal(size=size).astype(np.float32)
        output_values = np.empty(size // 2 + 1, np.complex64)
    elif operation == "irfft":
        input_values = (
            rng.normal(size=size // 2 + 1)
            + 1j * rng.normal(size=size // 2 + 1)
        ).astype(np.complex64)
        output_values = np.empty(size, np.float32)
    else:
        input_values = (
            rng.normal(size=size) + 1j * rng.normal(size=size)
        ).astype(np.complex64)
        output_values = np.empty(size, np.complex64)

    with scheme.TorchScheme("cpu"):
        input_array = Array(input_values)
        output_array = Array(output_values)
        original_transform = getattr(torch.fft, operation)
        expected = original_transform(
            input_array._data.tensor,
            n=size,
            norm="forward" if inverse else None,
        )
        direct_outputs = []
        normalizations = []

        def record_transform(input_tensor, n=None, dim=-1, norm=None, *, out=None):
            direct_outputs.append(out)
            normalizations.append(norm)
            return original_transform(
                input_tensor, n=n, dim=dim, norm=norm, out=out
            )

        monkeypatch.setattr(
            getattr(torchfft.torch, "fft"), operation, record_transform
        )
        transform = torchfft.ifft if inverse else torchfft.fft
        transform(input_array, output_array, None, itype, otype)

        assert direct_outputs == [output_array._data.tensor]
        assert normalizations == (["forward"] if inverse else [None])
        assert torch.equal(output_array._data.tensor, expected)


@pytest.mark.parametrize(("operation", "inverse"), [("fft", False), ("ifft", True)])
@pytest.mark.parametrize("partial_overlap", (False, True))
def test_aliased_complex_fft_preserves_allocation_before_copy(
    monkeypatch, operation, inverse, partial_overlap
):
    from pycbc.fft import torchfft

    rng = np.random.default_rng(2701)
    values = (rng.normal(size=32) + 1j * rng.normal(size=32)).astype(
        np.complex64
    )

    with scheme.TorchScheme("cpu"):
        if partial_overlap:
            storage = Array(np.append(values, np.complex64(0)))
            input_array = storage[:-1]
            output_array = storage[1:]
        else:
            input_array = Array(values)
            output_array = input_array
        original_transform = getattr(torch.fft, operation)
        expected = original_transform(
            input_array._data.tensor.clone(),
            n=len(input_array),
            norm="forward" if inverse else None,
        )
        direct_outputs = []
        normalizations = []

        def record_transform(input_tensor, n=None, dim=-1, norm=None, *, out=None):
            direct_outputs.append(out)
            normalizations.append(norm)
            if out is None:
                return original_transform(
                    input_tensor, n=n, dim=dim, norm=norm
                )
            return original_transform(
                input_tensor, n=n, dim=dim, norm=norm, out=out
            )

        monkeypatch.setattr(
            getattr(torchfft.torch, "fft"), operation, record_transform
        )
        transform = torchfft.ifft if inverse else torchfft.fft
        transform(input_array, output_array, None, "complex", "complex")

        assert direct_outputs == [None]
        assert normalizations == (["forward"] if inverse else [None])
        assert torch.equal(output_array._data.tensor, expected)


@pytest.mark.parametrize("size", (30, 32))
@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize(
    ("operation", "otype"),
    (("ifft", "complex"), ("irfft", "real")),
)
def test_ifft_preserves_unnormalized_contract(size, dtype, operation, otype):
    """The optimized inverse path retains PyCBC's backend normalization."""
    from pycbc.fft import torchfft

    rng = np.random.default_rng(3701)
    input_size = size if operation == "ifft" else size // 2 + 1
    input_values = (
        rng.normal(size=input_size) + 1j * rng.normal(size=input_size)
    ).astype(dtype)
    output_dtype = dtype if otype == "complex" else np.empty((), dtype).real.dtype
    output_values = np.empty(size, output_dtype)
    if operation == "ifft":
        expected = np.fft.ifft(input_values, n=size) * size
    else:
        expected = np.fft.irfft(input_values, n=size) * size

    with scheme.TorchScheme("cpu"):
        input_array = Array(input_values)
        output_array = Array(output_values)
        torchfft.ifft(input_array, output_array, None, "complex", otype)
        actual = output_array._data.tensor.detach().cpu().numpy().copy()

    tolerance = 2e-6 if dtype == np.complex64 else 1e-12
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize(
    "values",
    [
        np.array([-3.0, 0.5, 2.0, -1.0], dtype=np.float32),
        np.array([1 + 2j, -3 + 0.5j, 0.25 - 4j], dtype=np.complex64),
        np.array([1 + 2j, -3 + 0.5j, 0.25 - 4j], dtype=np.complex128),
    ],
)
def test_threshold_magnitude_squared_avoids_sqrt(values, monkeypatch):
    from pycbc.events import threshold_torch

    with scheme.TorchScheme("cpu"):
        tensor = Array(values)._data.tensor
        expected = (
            tensor.real.square() + tensor.imag.square()
            if tensor.is_complex()
            else tensor.square()
        )

        def fail_abs(*args, **kwargs):
            raise AssertionError("magnitude squared computed an unused square root")

        monkeypatch.setattr(threshold_torch.torch, "abs", fail_abs)
        actual = threshold_torch._magnitude_squared(tensor)

    assert actual.dtype == tensor.real.dtype
    assert torch.equal(actual, expected)


def test_threshold_compile_environment_gates(monkeypatch):
    from pycbc.events import threshold_torch

    names = (
        "PYCBC_TORCH_COMPILE",
        "PYCBC_TORCH_COMPILE_THRESHOLD",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    assert not threshold_torch._threshold_compile_requested()

    monkeypatch.setenv("PYCBC_TORCH_COMPILE", "yes")
    assert threshold_torch._threshold_compile_requested()
    monkeypatch.setenv("PYCBC_TORCH_COMPILE_THRESHOLD", "off")
    assert not threshold_torch._threshold_compile_requested()

    # A component switch cannot bypass the default-off master gate.
    monkeypatch.setenv("PYCBC_TORCH_COMPILE", "0")
    monkeypatch.setenv("PYCBC_TORCH_COMPILE_THRESHOLD", "1")
    assert not threshold_torch._threshold_compile_requested()

    monkeypatch.setenv("PYCBC_TORCH_COMPILE", "invalid")
    with pytest.raises(ValueError, match="PYCBC_TORCH_COMPILE"):
        threshold_torch._threshold_compile_requested()

    monkeypatch.setenv("PYCBC_TORCH_COMPILE", "1")
    monkeypatch.setenv("PYCBC_TORCH_COMPILE_THRESHOLD", "invalid")
    with pytest.raises(ValueError, match="PYCBC_TORCH_COMPILE_THRESHOLD"):
        threshold_torch._threshold_compile_requested()


def test_threshold_compile_gate_is_cuda_single_series_inference_only(
    monkeypatch,
):
    from pycbc.events import threshold_torch

    monkeypatch.setenv("PYCBC_TORCH_COMPILE", "1")
    monkeypatch.delenv("PYCBC_TORCH_COMPILE_THRESHOLD", raising=False)
    monkeypatch.setattr(
        threshold_torch, "_has_forward_ad_state", lambda tensor: False
    )
    # The structural CUDA matrix uses lightweight stand-ins so it can run on
    # CPU-only CI. Exact Tensor/layout/inference qualification is exercised
    # independently below.
    monkeypatch.setattr(
        threshold_torch,
        "_plain_strided_compile_tensor",
        lambda tensor: True,
    )

    def candidate(
        *,
        device="cuda",
        dtype=torch.complex64,
        ndim=1,
        size=64,
        requires_grad=False,
        contiguous=True,
        conjugated=False,
        negative=False,
    ):
        real_dtype = (
            torch.float32 if dtype == torch.complex64 else torch.float64
        )
        return SimpleNamespace(
            device=SimpleNamespace(type=device, index=0),
            dtype=dtype,
            real=SimpleNamespace(dtype=real_dtype),
            ndim=ndim,
            requires_grad=requires_grad,
            numel=lambda: size,
            is_contiguous=lambda: contiguous,
            is_conj=lambda: conjugated,
            is_neg=lambda: negative,
        )

    def threshold(
        *,
        device="cuda",
        dtype=torch.float32,
        ndim=0,
        requires_grad=False,
    ):
        return SimpleNamespace(
            device=SimpleNamespace(type=device, index=0),
            dtype=dtype,
            ndim=ndim,
            requires_grad=requires_grad,
        )

    assert threshold_torch._threshold_compile_eligible(
        candidate(), threshold(), single_series=True
    )
    assert not threshold_torch._threshold_compile_eligible(
        candidate(device="cpu"), threshold(), single_series=True
    )
    assert not threshold_torch._threshold_compile_eligible(
        candidate(dtype=torch.complex128), threshold(), single_series=True
    )
    assert not threshold_torch._threshold_compile_eligible(
        candidate(ndim=2), threshold(), single_series=True
    )
    assert not threshold_torch._threshold_compile_eligible(
        candidate(size=0), threshold(), single_series=True
    )
    assert not threshold_torch._threshold_compile_eligible(
        candidate(requires_grad=True), threshold(), single_series=True
    )
    assert not threshold_torch._threshold_compile_eligible(
        candidate(contiguous=False), threshold(), single_series=True
    )
    assert not threshold_torch._threshold_compile_eligible(
        candidate(conjugated=True), threshold(), single_series=True
    )
    assert not threshold_torch._threshold_compile_eligible(
        candidate(negative=True), threshold(), single_series=True
    )
    assert not threshold_torch._threshold_compile_eligible(
        candidate(), threshold(), single_series=False
    )
    assert not threshold_torch._threshold_compile_eligible(
        candidate(), threshold(device="cpu"), single_series=True
    )
    assert not threshold_torch._threshold_compile_eligible(
        candidate(), threshold(dtype=torch.float64), single_series=True
    )
    assert not threshold_torch._threshold_compile_eligible(
        candidate(), threshold(ndim=1), single_series=True
    )
    assert not threshold_torch._threshold_compile_eligible(
        candidate(), threshold(requires_grad=True), single_series=True
    )

    monkeypatch.setattr(
        threshold_torch, "_has_forward_ad_state", lambda tensor: True
    )
    assert not threshold_torch._threshold_compile_eligible(
        candidate(), threshold(), single_series=True
    )


def test_threshold_compile_requires_plain_strided_noninference_tensors():
    from pycbc.events import threshold_torch

    ordinary = torch.ones(4, dtype=torch.complex64)

    class DispatchTensor(torch.Tensor):
        pass

    subclass = torch.Tensor._make_subclass(
        DispatchTensor, ordinary, ordinary.requires_grad
    )
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0]]), torch.tensor([1.0]), size=(4,)
    )

    assert threshold_torch._plain_strided_compile_tensor(ordinary)
    assert not threshold_torch._plain_strided_compile_tensor(subclass)
    assert not threshold_torch._plain_strided_compile_tensor(sparse)

    if threshold_torch._TORCH_IS_INFERENCE is not None:
        with torch.inference_mode():
            inference = torch.ones(4, dtype=torch.complex64)
        assert not threshold_torch._plain_strided_compile_tensor(inference)


def test_threshold_compile_cache_is_bounded_and_exact(monkeypatch):
    from pycbc.events import threshold_torch

    compile_calls = []

    def fake_compile(function, **configuration):
        compile_calls.append(configuration)
        return function

    monkeypatch.setattr(threshold_torch.torch, "compile", fake_compile)
    cache = threshold_torch._compiled_threshold_core
    cache.cache_clear()
    try:
        base = (64, 4, "cuda", 0, torch.complex64, "inductor", "default")
        first = cache(*base)
        assert cache(*base) is first
        assert len(compile_calls) == 1
        assert compile_calls[0] == {
            "backend": "inductor",
            "mode": "default",
            "fullgraph": True,
            "dynamic": False,
        }

        variants = (
            (65, 4, "cuda", 0, torch.complex64, "inductor", "default"),
            (64, 8, "cuda", 0, torch.complex64, "inductor", "default"),
            (64, 4, "cpu", 0, torch.complex64, "inductor", "default"),
            (64, 4, "cuda", 1, torch.complex64, "inductor", "default"),
            (64, 4, "cuda", 0, torch.complex128, "inductor", "default"),
            (64, 4, "cuda", 0, torch.complex64, "eager", "default"),
            (
                64,
                4,
                "cuda",
                0,
                torch.complex64,
                "inductor",
                "reduce-overhead",
            ),
        )
        for variant in variants:
            assert cache(*variant) is not first
        assert len(compile_calls) == 1 + len(variants)

        cache.cache_clear()
        for length in range(
            1, threshold_torch._THRESHOLD_COMPILE_CACHE_SIZE + 2
        ):
            cache(
                length,
                4,
                "cuda",
                0,
                torch.complex64,
                "inductor",
                "default",
            )
        assert (
            cache.cache_info().currsize
            == threshold_torch._THRESHOLD_COMPILE_CACHE_SIZE
        )
    finally:
        cache.cache_clear()


def test_threshold_compile_dispatch_verifies_raw_outputs_without_inductor(
    monkeypatch,
):
    from pycbc.events import threshold_torch

    tensor = torch.tensor([1, 3, 2, 0, 4], dtype=torch.complex64)
    observed_keys = []

    monkeypatch.setattr(
        threshold_torch,
        "_threshold_compile_eligible",
        lambda tensor, threshold_sq, *, single_series: single_series,
    )
    monkeypatch.setattr(
        threshold_torch,
        "_threshold_compile_configuration",
        lambda: ("test-backend", "test-mode"),
    )
    monkeypatch.setenv("PYCBC_TORCH_COMPILE_VERIFY", "1")

    def fake_cached_core(*key):
        observed_keys.append(key)
        return lambda value, threshold_sq: (
            threshold_torch._fixed_shape_threshold_core(
                value, threshold_sq, 2
            )
        )

    monkeypatch.setattr(
        threshold_torch, "_compiled_threshold_core", fake_cached_core
    )
    actual = threshold_torch._run_threshold_core(
        tensor, torch.tensor(1.0), 2, single_series=True
    )
    expected = threshold_torch._fixed_shape_threshold_core(
        tensor, torch.tensor(1.0), 2
    )

    assert observed_keys == [
        (
            tensor.numel(),
            2,
            "cpu",
            None,
            torch.complex64,
            "test-backend",
            "test-mode",
        )
    ]
    assert all(torch.equal(a, e) for a, e in zip(actual, expected))


@pytest.mark.parametrize("window", (0, -1))
def test_threshold_compile_dispatch_preserves_invalid_window_error(
    monkeypatch, window
):
    from pycbc.events import threshold_torch

    def unexpected_dispatch(*args, **kwargs):
        raise AssertionError("invalid windows must fail before dispatch")

    monkeypatch.setattr(
        threshold_torch, "_threshold_compile_eligible", unexpected_dispatch
    )
    with pytest.raises(ValueError, match="window must be positive"):
        threshold_torch._run_threshold_core(
            torch.ones(4, dtype=torch.complex64),
            torch.tensor(1.0),
            window,
            single_series=True,
        )


@pytest.mark.parametrize("window", (0, -1))
def test_threshold_and_cluster_preserves_invalid_window_error(window):
    from pycbc.events import threshold_torch

    # The eager implementation historically validates its window before
    # attempting threshold conversion. Keep that public error contract while
    # thresholding is refactored around the optional compiled core.
    with pytest.raises(ValueError, match="window must be positive"):
        threshold_torch.threshold_and_cluster(
            torch.ones(4, dtype=torch.complex64), object(), window
        )


def test_threshold_compile_mismatch_and_errors_do_not_fall_back(monkeypatch):
    from pycbc.events import threshold_torch

    tensor = torch.tensor([1, 3, 2, 0], dtype=torch.complex64)
    monkeypatch.setattr(
        threshold_torch,
        "_threshold_compile_eligible",
        lambda tensor, threshold_sq, *, single_series: True,
    )
    monkeypatch.setattr(
        threshold_torch,
        "_threshold_compile_configuration",
        lambda: ("test-backend", "test-mode"),
    )
    monkeypatch.setenv("PYCBC_TORCH_COMPILE_VERIFY", "1")

    corruptions = (
        (
            "block maxima differ",
            lambda maxima, indices, keep: (
                maxima.clone().add_(1),
                indices,
                keep,
            ),
        ),
        (
            "block indices differ",
            lambda maxima, indices, keep: (
                maxima,
                indices.clone().add_(1),
                keep,
            ),
        ),
        (
            "survivor mask differs",
            lambda maxima, indices, keep: (
                maxima,
                indices,
                keep.clone().logical_not_(),
            ),
        ),
    )
    for message, corrupt in corruptions:
        def mismatching_core(*key):
            def run(value, threshold_sq):
                outputs = threshold_torch._fixed_shape_threshold_core(
                    value, threshold_sq, 2
                )
                return corrupt(*outputs)

            return run

        monkeypatch.setattr(
            threshold_torch, "_compiled_threshold_core", mismatching_core
        )
        with pytest.raises(RuntimeError, match=message):
            threshold_torch._run_threshold_core(
                tensor, torch.tensor(1.0), 2, single_series=True
            )

    def failing_core(*key):
        def run(value, threshold_sq):
            raise RuntimeError("compiler execution failed")

        return run

    monkeypatch.setattr(
        threshold_torch, "_compiled_threshold_core", failing_core
    )
    with pytest.raises(RuntimeError, match="compiler execution failed"):
        threshold_torch._run_threshold_core(
            tensor, torch.tensor(1.0), 2, single_series=True
        )

    assert not threshold_torch._tensor_bitwise_equal(
        torch.tensor([0.0]), torch.tensor([-0.0])
    )


def test_threshold_clustering_reuses_magnitude_without_threshold_mask(monkeypatch):
    from pycbc.events import threshold_torch

    values = np.array([1, 6, 2, 1, 4, 1], np.complex64)
    with scheme.TorchScheme("cpu"):
        series = Array(values)
        expected_magnitude_squared = (
            series._data.tensor.real.square()
            + series._data.tensor.imag.square()
        )
        assert torch.equal(
            threshold_torch._magnitude_squared(series._data.tensor),
            expected_magnitude_squared,
        )

        def fail_mask(*args, **kwargs):
            raise AssertionError("clustering built an unused threshold mask")

        monkeypatch.setattr(threshold_torch, "_threshold_mask", fail_mask)
        function_values, function_indices = threshold_torch.threshold_and_cluster(
            series, 3.0, 2
        )
        engine_values, engine_indices = threshold_torch.TorchThresholdCluster(
            series
        ).threshold_and_cluster(3.0, 2)

        for result in (
            function_values,
            function_indices,
            engine_values,
            engine_indices,
        ):
            assert isinstance(result._data, TorchArrayData)
        np.testing.assert_array_equal(function_indices.numpy(), [1, 4])
        np.testing.assert_array_equal(engine_indices.numpy(), [1, 4])
        np.testing.assert_array_equal(function_values.numpy(), values[[1, 4]])
        np.testing.assert_array_equal(engine_values.numpy(), values[[1, 4]])


def test_threshold_engine_reuses_cpu_magnitude_workspace(monkeypatch):
    from pycbc.events import threshold_torch

    values = np.linspace(-3.0, 4.0, 32, dtype=np.float32)
    calls = []
    original = threshold_torch._magnitude_squared

    def record(tensor, out=None, component_out=None):
        calls.append((out, component_out))
        return original(tensor, out=out, component_out=component_out)

    monkeypatch.setattr(threshold_torch, "_magnitude_squared", record)
    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(Array(values))
        expected_np = values.real ** 2
        expected_magnitude = torch.from_numpy(expected_np)
        first_values, first_indices = engine.threshold_and_cluster(1.0, 4)
        assert torch.equal(engine._magnitude, expected_magnitude)
        second_values, second_indices = engine.threshold_and_cluster(1.0, 4)

    assert len(calls) == 2
    assert all(call[0] is engine._magnitude for call in calls)
    assert all(call[1] is engine._magnitude_component for call in calls)
    assert torch.equal(first_values._data.tensor, second_values._data.tensor)
    assert torch.equal(first_indices._data.tensor, second_indices._data.tensor)


def test_threshold_engine_native_cpu_path_is_zero_copy_and_results_are_stable(
    monkeypatch,
):
    from pycbc.events import threshold_torch

    initial = np.array([0, 3, 0, 1], dtype=np.complex64)
    overwritten = np.array([0, 1, 0, 4], dtype=np.complex64)
    native_inputs = []

    def fake_native(
        series, slen, values, locs, threshold, window, segsize, *args
    ):
        native_inputs.append(
            (series.__array_interface__["data"][0], series.copy())
        )
        index = int(np.argmax(series.real**2 + series.imag**2))
        values[0] = series[index]
        locs[0] = index
        return 1

    monkeypatch.setattr(
        threshold_torch, "parallel_thresh_cluster", fake_native
    )
    with scheme.TorchScheme("cpu"):
        series = Array(initial)
        source = series._data.tensor
        engine = threshold_torch.TorchThresholdCluster(series)
        assert engine._magnitude is None
        assert engine._magnitude_component is None
        first_values, first_indices = engine.threshold_and_cluster(0.5, 2)
        first_native_series = engine._native_series
        source.copy_(torch.from_numpy(overwritten))
        second_values, second_indices = engine.threshold_and_cluster(0.5, 2)

    assert len(native_inputs) == 2
    assert all(item[0] == source.data_ptr() for item in native_inputs)
    assert engine._native_series is first_native_series
    np.testing.assert_array_equal(native_inputs[0][1], initial)
    np.testing.assert_array_equal(native_inputs[1][1], overwritten)
    assert isinstance(first_values._data, TorchArrayData)
    assert isinstance(first_indices._data, TorchArrayData)
    assert first_indices._data.tensor.dtype == torch.int64
    np.testing.assert_array_equal(first_indices.numpy(), [1])
    np.testing.assert_array_equal(first_values.numpy(), initial[[1]])
    np.testing.assert_array_equal(second_indices.numpy(), [3])
    np.testing.assert_array_equal(second_values.numpy(), overwritten[[3]])


def test_threshold_engine_native_cpu_input_tracks_resize_and_storage_changes(
    monkeypatch,
):
    from pycbc.events import threshold_torch

    native_inputs = []

    def fake_native(
        series, slen, values, locs, threshold, window, segsize, *args
    ):
        native_inputs.append(
            (series.__array_interface__["data"][0], series.copy())
        )
        index = int(np.argmax(series[:slen].real))
        values[0] = series[index]
        locs[0] = index
        return 1

    monkeypatch.setattr(
        threshold_torch, "parallel_thresh_cluster", fake_native
    )
    initial = torch.tensor([0, 3, 0, 1], dtype=torch.complex64)
    replacement = torch.tensor([0, 1, 0, 4], dtype=torch.complex64)
    resized = torch.tensor([0, 1, 0, 2, 0, 5], dtype=torch.complex64)
    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(initial)
        first_values, first_indices = engine.threshold_and_cluster(0.5, 2)
        first_native_series = engine._native_series
        initial.resize_(2)
        second_values, second_indices = engine.threshold_and_cluster(0.5, 2)
        resized_native_series = engine._native_series
        initial.set_(replacement)
        third_values, third_indices = engine.threshold_and_cluster(0.5, 2)
        replacement_native_series = engine._native_series
        initial.set_(resized)
        fourth_values, fourth_indices = engine.threshold_and_cluster(0.5, 2)
        final_native_series = engine._native_series

    assert len(native_inputs) == 4
    assert resized_native_series is not first_native_series
    assert replacement_native_series is not resized_native_series
    assert final_native_series is not replacement_native_series
    assert native_inputs[0][0] == native_inputs[1][0]
    assert native_inputs[1][0] != native_inputs[2][0]
    assert native_inputs[2][0] != native_inputs[3][0]
    assert [item[1].size for item in native_inputs] == [4, 2, 4, 6]
    np.testing.assert_array_equal(first_indices.numpy(), [1])
    np.testing.assert_array_equal(first_values.numpy(), [3])
    np.testing.assert_array_equal(second_indices.numpy(), [1])
    np.testing.assert_array_equal(second_values.numpy(), [3])
    np.testing.assert_array_equal(third_indices.numpy(), [3])
    np.testing.assert_array_equal(third_values.numpy(), [4])
    np.testing.assert_array_equal(fourth_indices.numpy(), [5])
    np.testing.assert_array_equal(fourth_values.numpy(), [5])


def test_threshold_engine_native_cpu_input_flattens_non_1d_source():
    from pycbc.events import threshold_torch

    source = torch.tensor(
        [[0, 3, 0, 1], [0, 2, 0, 4]], dtype=torch.complex64
    )
    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(source)
        first_values, first_indices = engine.threshold_and_cluster(0.5, 2)
        source[1, 3] = 5
        second_values, second_indices = engine.threshold_and_cluster(0.5, 2)

    assert engine.series.ndim == 1
    np.testing.assert_array_equal(first_indices.numpy(), [1, 7])
    np.testing.assert_array_equal(first_values.numpy(), [3, 4])
    np.testing.assert_array_equal(second_indices.numpy(), [1, 7])
    np.testing.assert_array_equal(second_values.numpy(), [3, 5])


def test_threshold_engine_conjugate_view_uses_torch_fallback():
    from pycbc.events import threshold_torch

    source = torch.tensor([0, 3 + 4j, 0, 2 + 1j]).conj()
    assert source.is_conj()
    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(source)
        assert engine._magnitude is not None
        selected, indices = engine.threshold_and_cluster(1.0, 2)

    torch.testing.assert_close(indices._data.tensor, torch.tensor([1]))
    torch.testing.assert_close(selected._data.tensor, source[[1]])


def test_threshold_engine_tensor_subclass_uses_torch_fallback(monkeypatch):
    from pycbc.events import threshold_torch

    class DispatchTensor(torch.Tensor):
        pass

    base = torch.tensor(
        [0, 3 + 4j, 0, 2 + 1j], dtype=torch.complex64
    )
    source = torch.Tensor._make_subclass(
        DispatchTensor, base, base.requires_grad
    )

    def fail_native(*args, **kwargs):
        raise AssertionError("Tensor subclass entered native CPU path")

    monkeypatch.setattr(
        threshold_torch, "parallel_thresh_cluster", fail_native
    )
    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(source)
        assert not engine._can_use_native_cpu()
        selected, indices = engine.threshold_and_cluster(1.0, 2)

    torch.testing.assert_close(indices._data.tensor, torch.tensor([1]))
    torch.testing.assert_close(selected._data.tensor, source[[1]])


def test_threshold_engine_tensor_subclass_never_uses_out_scratch(monkeypatch):
    from pycbc.events import threshold_torch

    class RejectOutTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            if kwargs.get("out") is not None:
                raise AssertionError("Tensor subclass entered an out= kernel")
            return super().__torch_function__(func, types, args, kwargs)

    base = torch.tensor(
        [0, 3 + 4j, 0, 2 + 1j], dtype=torch.complex64
    )
    source = torch.Tensor._make_subclass(
        RejectOutTensor, base, base.requires_grad
    )

    def fail_native(*args, **kwargs):
        raise AssertionError("Tensor subclass entered native CPU path")

    monkeypatch.setattr(
        threshold_torch, "parallel_thresh_cluster", fail_native
    )
    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(source)
        assert engine._magnitude is None
        assert engine._magnitude_component is None
        selected, indices = engine.threshold_and_cluster(1.0, 2)

    torch.testing.assert_close(indices._data.tensor, torch.tensor([1]))
    torch.testing.assert_close(selected._data.tensor, source[[1]])


def test_threshold_engine_stale_scratch_falls_back_after_data_rebind():
    from pycbc.events import threshold_torch

    source = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(source)
        stale_magnitude = engine._magnitude
        assert stale_magnitude is not None

        # Change both shape and dtype while retaining the bound Tensor object.
        # A stale float32 out= square would overflow here and invent a survivor.
        source.data = torch.tensor([1e20, 0.0], dtype=torch.float64)
        expected_values, expected_indices = (
            threshold_torch.threshold_and_cluster(source, 1e21, 2)
        )
        actual_values, actual_indices = engine.threshold_and_cluster(1e21, 2)

    assert engine._magnitude is stale_magnitude
    assert actual_values._data.tensor.dtype == torch.float64
    assert torch.equal(
        actual_values._data.tensor.view(torch.uint8),
        expected_values._data.tensor.view(torch.uint8),
    )
    assert torch.equal(
        actual_indices._data.tensor.view(torch.uint8),
        expected_indices._data.tensor.view(torch.uint8),
    )


def test_threshold_engine_inference_tensor_uses_torch_fallback(monkeypatch):
    from pycbc.events import threshold_torch

    if threshold_torch._TORCH_IS_INFERENCE is None:
        pytest.skip("this Torch version cannot identify inference tensors")

    def fail_native(*args, **kwargs):
        raise AssertionError("inference tensor entered native CPU path")

    monkeypatch.setattr(
        threshold_torch, "parallel_thresh_cluster", fail_native
    )
    with scheme.TorchScheme("cpu"):
        with torch.inference_mode():
            source = torch.tensor(
                [0, 3 + 4j, 0, 2 + 1j], dtype=torch.complex64
            )
            engine = threshold_torch.TorchThresholdCluster(source)
            assert not engine._can_use_native_cpu()
            assert engine._magnitude is None
            assert engine._magnitude_component is None
        # Reusing the engine after leaving inference mode must not try to
        # write to scratch tensors created under the prior ambient mode.
        selected, indices = engine.threshold_and_cluster(1.0, 2)

    torch.testing.assert_close(indices._data.tensor, torch.tensor([1]))
    torch.testing.assert_close(selected._data.tensor, source[[1]])


def test_raw_snr_threshold_preserves_strict_ties_and_rejects_nan():
    """Raw threshold/norm has the production strict-``>`` edge behavior."""
    from pycbc.events import threshold_torch

    boundary = np.float32(4.0)
    above = np.nextafter(boundary, np.float32(np.inf), dtype=np.float32)
    values = np.array([boundary, above, np.nan], dtype=np.complex64)

    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(Array(values))
        selected, indices = engine.threshold_and_cluster(2.0 / 0.5, 1)

    np.testing.assert_array_equal(indices.numpy(), [1])
    np.testing.assert_array_equal(selected.numpy(), values[[1]])


def test_threshold_engine_ignores_nan_beside_valid_block_maximum():
    from pycbc.events import threshold_torch

    values = np.array([3 + 0j, np.nan + 0j, 0, 0], dtype=np.complex64)
    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(Array(values))
        selected, indices = engine.threshold_and_cluster(1.0, 2)

    np.testing.assert_array_equal(indices.numpy(), [0])
    np.testing.assert_array_equal(selected.numpy(), values[[0]])


def test_threshold_engine_reads_overwritten_series_workspace():
    from pycbc.events import threshold_torch

    initial = np.array([0, 3, 0, 1], dtype=np.complex64)
    overwritten = np.array([0, 1, 0, 4], dtype=np.complex64)
    with scheme.TorchScheme("cpu"):
        series = Array(initial)
        engine = threshold_torch.TorchThresholdCluster(series)
        first_values, first_indices = engine.threshold_and_cluster(0.5, 2)
        series._data.tensor.copy_(torch.from_numpy(overwritten))
        second_values, second_indices = engine.threshold_and_cluster(0.5, 2)

    np.testing.assert_array_equal(first_indices.numpy(), [1])
    np.testing.assert_array_equal(first_values.numpy(), initial[[1]])
    np.testing.assert_array_equal(second_indices.numpy(), [3])
    np.testing.assert_array_equal(second_values.numpy(), overwritten[[3]])


@pytest.mark.parametrize(
    ("values", "expected_indices"),
    [
        (np.empty(0, dtype=np.complex64), np.empty(0, dtype=np.int64)),
        (
            np.array([2, 0, 3, 0, 2], dtype=np.complex64),
            np.array([0, 2, 4], dtype=np.int64),
        ),
    ],
)
def test_threshold_engine_empty_and_all_survivor_inputs(
    values, expected_indices
):
    from pycbc.events import threshold_torch

    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(Array(values))
        selected, indices = engine.threshold_and_cluster(1.0, 1)

    np.testing.assert_array_equal(indices.numpy(), expected_indices)
    np.testing.assert_array_equal(selected.numpy(), values[expected_indices])


def test_threshold_engine_preserves_autograd_without_out_workspace():
    from pycbc.events import threshold_torch

    source = torch.tensor(
        [2 + 0j, 0, 3 + 0j, 0, 2 + 0j],
        dtype=torch.complex64,
        requires_grad=True,
    )
    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(source)
        assert engine._magnitude is None
        assert engine._magnitude_component is None
        selected, indices = engine.threshold_and_cluster(1.0, 1)
    selected._data.tensor.real.sum().backward()

    torch.testing.assert_close(indices._data.tensor, torch.arange(0, 5, 2))
    torch.testing.assert_close(
        source.grad,
        torch.tensor([1, 0, 1, 0, 1], dtype=torch.complex64),
    )


def test_threshold_engine_honors_requires_grad_enabled_after_construction():
    from pycbc.events import threshold_torch

    source = torch.tensor(
        [2 + 0j, 0, 3 + 0j, 0, 2 + 0j],
        dtype=torch.complex64,
    )
    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(source)
        # The no-grad complex64 engine starts on the native CPU fast path and
        # therefore does not allocate eager-Torch magnitude scratch. Enabling
        # gradients later must still switch safely to the pure Torch path.
        assert engine._magnitude is None
        source.requires_grad_(True)
        selected, indices = engine.threshold_and_cluster(1.0, 1)
    selected._data.tensor.real.sum().backward()

    torch.testing.assert_close(indices._data.tensor, torch.arange(0, 5, 2))
    torch.testing.assert_close(
        source.grad,
        torch.tensor([1, 0, 1, 0, 1], dtype=torch.complex64),
    )


def test_threshold_engine_preserves_forward_ad_without_native_export(
    monkeypatch,
):
    from pycbc.events import threshold_torch

    def fail_native(*args, **kwargs):
        raise AssertionError("forward-AD tensor was exported to NumPy")

    monkeypatch.setattr(
        threshold_torch, "parallel_thresh_cluster", fail_native
    )
    primal = torch.tensor(
        [2 + 0j, 0, 3 + 0j, 0, 2 + 0j], dtype=torch.complex64
    )
    tangent = torch.tensor(
        [1 + 2j, 0, 3 + 4j, 0, 5 + 6j], dtype=torch.complex64
    )
    with scheme.TorchScheme("cpu"):
        with torch.autograd.forward_ad.dual_level():
            source = torch.autograd.forward_ad.make_dual(primal, tangent)
            engine = threshold_torch.TorchThresholdCluster(source)
            assert engine._magnitude is None
            assert engine._magnitude_component is None
            selected, indices = engine.threshold_and_cluster(1.0, 1)
            selected_dual = torch.autograd.forward_ad.unpack_dual(
                selected._data.tensor
            )

            torch.testing.assert_close(
                indices._data.tensor, torch.arange(0, 5, 2)
            )
            torch.testing.assert_close(
                selected_dual.primal, primal[[0, 2, 4]]
            )
            torch.testing.assert_close(
                selected_dual.tangent, tangent[[0, 2, 4]]
            )


def test_threshold_symmetric_cluster_reuses_threshold_mask(monkeypatch):
    from pycbc.events import threshold_torch

    max_values = torch.tensor(
        [16.0, 36.0, 36.0, 9.0, 49.0, 64.0, 4.0]
    )
    max_indices = torch.arange(len(max_values), dtype=torch.long)

    def fail_scratch(*args, **kwargs):
        raise AssertionError("clustering allocated a neighbor scratch buffer")

    monkeypatch.setattr(threshold_torch.torch, "full", fail_scratch)
    monkeypatch.setattr(threshold_torch.torch, "zeros", fail_scratch)
    values, indices = threshold_torch._symmetric_cluster(
        max_values, max_indices, torch.tensor(25.0)
    )

    # The legacy asymmetric tie rule keeps the leftmost interior maximum.
    assert torch.equal(values, max_values[[1, 5]])
    assert torch.equal(indices, max_indices[[1, 5]])


def test_threshold_symmetric_cluster_matches_legacy_nan_comparisons():
    from pycbc.events import threshold_torch

    max_values = torch.tensor([36.0, float("nan"), 49.0])
    max_indices = torch.arange(len(max_values), dtype=torch.long)

    values, indices = threshold_torch._symmetric_cluster(
        max_values, max_indices, torch.tensor(25.0)
    )

    assert values.numel() == 0
    assert indices.numel() == 0


@pytest.mark.parametrize(
    ("values", "expected_indices"),
    [
        (
            np.array([0, 0, 3, 0, 3, 0, 0, 0], dtype=np.complex64),
            np.array([2], dtype=np.int64),
        ),
        (
            np.array([3, 0, 3, 0], dtype=np.complex64),
            np.empty(0, dtype=np.int64),
        ),
    ],
)
def test_threshold_engine_matches_legacy_adjacent_block_ties(
    values, expected_indices
):
    from pycbc.events import threshold_torch

    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(Array(values))
        selected, indices = engine.threshold_and_cluster(1.0, 2)

    np.testing.assert_array_equal(indices.numpy(), expected_indices)
    np.testing.assert_array_equal(selected.numpy(), values[expected_indices])


def test_threshold_cluster_candidates_return_global_indices_with_padding():
    from pycbc.events import threshold_torch

    magnitudes = torch.tensor([1.0, 9.0, 2.0, 3.0, 4.0])
    values, indices = threshold_torch._cluster_candidates(magnitudes, 2)

    assert torch.equal(values, torch.tensor([9.0, 3.0, 4.0]))
    assert torch.equal(indices, torch.tensor([1, 3, 4]))


def test_empty_point_chisq_skips_bin_device_work(monkeypatch):
    from pycbc.vetoes import chisq_torch

    with scheme.TorchScheme("cpu"):
        corr = FrequencySeries(
            np.ones(32, dtype=np.complex64), delta_f=0.125
        )
        points = Array(np.empty(0, dtype=np.int64))

        def fail_arange(*args, **kwargs):
            raise AssertionError("empty-point chisq launched bin work")

        monkeypatch.setattr(chisq_torch.torch, "arange", fail_arange)
        result = chisq_torch.shift_sum(
            corr, points, np.array([1, 4, 8, 16], dtype=np.int64)
        )

        assert isinstance(result._data, TorchArrayData)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.float32
        assert result._data.tensor.numel() == 0


def test_time_series_fft_overwrites_uninitialized_output(monkeypatch):
    from pycbc.filter import matchedfilter

    rng = np.random.default_rng(3701)
    values = rng.normal(size=64).astype(np.float32)

    with scheme.TorchScheme("cpu"):
        series = TimeSeries(values, delta_t=1 / 1024)
        original_empty = matchedfilter.empty
        allocations = []

        def nan_filled_empty(length, dtype):
            output = original_empty(length, dtype=dtype)
            output[:] = np.nan + 1j * np.nan
            allocations.append(output)
            return output

        monkeypatch.setattr(matchedfilter, "empty", nan_filled_empty)
        transformed = matchedfilter.make_frequency_series(series)

        assert len(allocations) == 1
        assert allocations[0]._data is transformed._data
        np.testing.assert_allclose(
            transformed.numpy(),
            np.fft.rfft(values) * series.delta_t,
            rtol=1e-6,
            atol=1e-6,
        )
        assert np.isfinite(transformed.numpy()).all()


def test_matched_filter_ifft_overwrites_uninitialized_output(monkeypatch):
    from pycbc.filter import matchedfilter

    rng = np.random.default_rng(4701)
    size = 64
    template_values = (
        rng.normal(size=size // 2 + 1)
        + 1j * rng.normal(size=size // 2 + 1)
    ).astype(np.complex64)
    data_values = (
        rng.normal(size=size // 2 + 1)
        + 1j * rng.normal(size=size // 2 + 1)
    ).astype(np.complex64)

    with scheme.TorchScheme("cpu"):
        template = FrequencySeries(template_values, delta_f=0.25)
        data = FrequencySeries(data_values, delta_f=0.25)
        expected_output = Array(np.zeros(size, dtype=np.complex64))
        expected, _, _ = matchedfilter.matched_filter_core(
            template, data, low_frequency_cutoff=0.25, h_norm=1.0,
            out=expected_output,
        )
        original_empty = matchedfilter.empty
        allocations = []

        def nan_filled_empty(length, dtype):
            output = original_empty(length, dtype=dtype)
            output[:] = np.nan + 1j * np.nan
            allocations.append(output)
            return output

        monkeypatch.setattr(matchedfilter, "empty", nan_filled_empty)
        actual, _, _ = matchedfilter.matched_filter_core(
            template, data, low_frequency_cutoff=0.25, h_norm=1.0,
        )

        assert len(allocations) == 1
        assert allocations[0]._data is actual._data
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        assert np.isfinite(actual.numpy()).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("storage_kind", ["protocol", "tensor_subclass"])
def test_search_public_storage_preserves_device_and_gradients(
    device, storage_kind,
):
    from pycbc.events import cuts, ranking, single, veto
    from pycbc.types.backend import backend_array

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    class PublicStorage:
        backend = "torch"

        def __init__(self, tensor):
            self.backend_array = tensor

    class ExternalTensor(torch.Tensor):
        pass

    source = torch.tensor([8.0, 12.0, 16.0], dtype=torch.float64,
                          device=device, requires_grad=True)
    reduced_chisq = torch.tensor(
        [1.0, 2.0, 4.0], dtype=torch.float64, device=device
    )
    if storage_kind == "protocol":
        values = PublicStorage(source)
        chisq = PublicStorage(reduced_chisq)
        tensor = source
    else:
        tensor = source.as_subclass(ExternalTensor)
        values = tensor
        chisq = reduced_chisq.as_subclass(ExternalTensor)
    with scheme.TorchScheme(device):
        for accessor in (cuts._torch_cut_tensor, single._torch_tensor,
                         veto._torch_veto_tensor):
            assert accessor(values) is tensor
        result = backend_array(ranking.newsnr(values, chisq))
        expected_weight = ((1 + reduced_chisq ** 3) / 2) ** (-1 / 6)
        torch.testing.assert_close(result, source * expected_weight)
        assert result.device == source.device
        result.sum().backward()
        torch.testing.assert_close(source.grad, expected_weight)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("storage_kind", ["protocol", "tensor"])
def test_event_thresholds_accept_public_backend_storage(
    device, storage_kind, monkeypatch,
):
    from pycbc.events import eventmgr

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    def storage(values):
        tensor = torch.tensor(values, dtype=torch.float32, device=device)
        if storage_kind == "protocol":
            return SimpleNamespace(backend="torch", backend_array=tensor)
        return tensor

    def reject_host_fallback(*args, **kwargs):
        raise AssertionError("Torch threshold used the host input path")

    monkeypatch.setattr(eventmgr, "threshold_real_numpy", reject_host_fallback)
    with scheme.TorchScheme(device):
        locations, values = eventmgr.threshold_real(
            storage([0.0, 1.0, 3.0, -2.0, 5.0]), 2.0
        )
        np.testing.assert_array_equal(locations, [2, 4])
        np.testing.assert_array_equal(values, [3.0, 5.0])
        positions = eventmgr.findchirp_cluster_over_window(
            storage([0.0, 1.0, 5.0]), storage([1.0, 3.0, 4.0]), 2
        )
        np.testing.assert_array_equal(positions, [1, 2])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_strain_overwhitening_preserves_storage_contract(device, monkeypatch):
    from types import SimpleNamespace
    from pycbc.strain.strain import StrainBuffer
    from pycbc.types.backend import backend_array, wrap_backend_array

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source = torch.linspace(0.0, 1.0, 64, dtype=torch.float64,
                            device=device, requires_grad=True)
    expected = torch.fft.rfft(source[-32:]) / 64.0
    with scheme.TorchScheme(device):
        strain = TimeSeries(wrap_backend_array(source), delta_t=1 / 32,
                            epoch=123.0, copy=False)
        psd = FrequencySeries(np.full(17, 2.0), delta_f=1.0)
        psd.psdt = psd
        buffer = SimpleNamespace(strain=strain, segments={}, sample_rate=32,
                                 reduced_pad=0, psds={1.0: psd})

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("strain overwhitening copied to the host")

        monkeypatch.setattr(Array, "numpy", reject_host_transfer)
        result = StrainBuffer.overwhitened_data(buffer, 1.0)
        tensor = backend_array(result, "torch")
        torch.testing.assert_close(tensor, expected)
        assert result.delta_f == 1.0
        assert float(result.epoch) == 124.0
        assert result.psd is psd
        assert StrainBuffer.overwhitened_data(buffer, 1.0) is result
        tensor.abs().square().sum().backward()
        assert source.grad is not None
        assert torch.isfinite(source.grad).all()
        assert source.grad[-32:].abs().sum() > 0
