"""Tests for opt-in trusted Torch CPU threshold-result construction."""

import numpy as np
import pytest
torch = pytest.importorskip("torch")


_GATE = "PYCBC_TORCH_CPU_THRESHOLD_TRUSTED_ARRAYS"


def _result_bytes(result):
    return result._data.tensor.detach().numpy().tobytes()


def test_trusted_threshold_result_gate_is_strict_and_latched(monkeypatch):
    import pycbc.scheme as pycbc_scheme
    from pycbc import scheme
    from pycbc.events import threshold_torch

    source = torch.ones(4, dtype=torch.complex64)
    assert threshold_torch._TRUSTED_NATIVE_RESULT_ENV == _GATE

    with scheme.TorchScheme("cpu"):
        monkeypatch.delenv(_GATE, raising=False)
        default_engine = threshold_torch.TorchThresholdCluster(source)
        assert default_engine._trusted_native_result_scheme is None

        monkeypatch.setenv(_GATE, "true")
        enabled_engine = threshold_torch.TorchThresholdCluster(source)
        assert (
            enabled_engine._trusted_native_result_scheme
            is pycbc_scheme.mgr.state
        )

        # The gate is intentionally read once per engine. A newly constructed
        # engine observes the new setting without adding an environment lookup
        # to every threshold call.
        monkeypatch.setenv(_GATE, "off")
        assert enabled_engine._trusted_native_result_scheme is not None
        disabled_engine = threshold_torch.TorchThresholdCluster(source)
        assert disabled_engine._trusted_native_result_scheme is None

        monkeypatch.setenv(_GATE, "sometimes")
        with pytest.raises(ValueError, match=_GATE):
            threshold_torch.TorchThresholdCluster(source)


def test_trusted_threshold_results_match_public_wrapper_contract(monkeypatch):
    import pycbc.scheme as pycbc_scheme
    from pycbc import scheme
    from pycbc.events import threshold_torch
    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData

    source = torch.tensor([2, 0, 3, 0, 2], dtype=torch.complex64)
    with scheme.TorchScheme("cpu"):
        monkeypatch.setenv(_GATE, "0")
        standard_engine = threshold_torch.TorchThresholdCluster(source)
        wrapper_calls = []
        standard_wrapper = threshold_torch._array_from_tensor

        def record_standard_wrapper(tensor):
            wrapper_calls.append(tensor)
            return standard_wrapper(tensor)

        with monkeypatch.context() as patch:
            patch.setattr(
                threshold_torch,
                "_array_from_tensor",
                record_standard_wrapper,
            )
            standard = standard_engine.threshold_and_cluster(1.0, 1)
        assert len(wrapper_calls) == 2

        monkeypatch.setenv(_GATE, "1")
        trusted_engine = threshold_torch.TorchThresholdCluster(source)

        def reject_standard_wrapper(_tensor):
            raise AssertionError("trusted route used the public constructor")

        with monkeypatch.context() as patch:
            patch.setattr(
                threshold_torch,
                "_array_from_tensor",
                reject_standard_wrapper,
            )
            trusted = trusted_engine.threshold_and_cluster(1.0, 1)

        expected_dtypes = (np.dtype(np.complex64), np.dtype(np.int64))
        expected_torch_dtypes = (torch.complex64, torch.int64)
        for reference, result, numpy_dtype, torch_dtype in zip(
            standard, trusted, expected_dtypes, expected_torch_dtypes
        ):
            assert type(result) is Array
            assert type(result._data) is TorchArrayData
            assert result._scheme is pycbc_scheme.mgr.state
            assert result._saved is None
            assert result.dtype == reference.dtype == numpy_dtype
            assert result._data.tensor.dtype == torch_dtype
            assert result._data.tensor.device.type == "cpu"
            assert result._data.tensor.ndim == 1
            assert result._data.tensor.is_contiguous()
            assert result.ptr == result._data.tensor.data_ptr()
            assert _result_bytes(result) == _result_bytes(reference)
            storage = result._data.tensor.untyped_storage()
            if hasattr(storage, "resizable"):
                assert not storage.resizable()

        np.testing.assert_array_equal(trusted[1].numpy(), [0, 2, 4])
        np.testing.assert_array_equal(trusted[0].numpy(), source.numpy()[::2])
        assert trusted[0].ptr != (
            trusted_engine._native_values.__array_interface__["data"][0]
        )
        assert trusted[1].ptr != (
            trusted_engine._native_indices.__array_interface__["data"][0]
        )


def test_trusted_threshold_results_are_fresh_and_stable(monkeypatch):
    from pycbc import scheme
    from pycbc.events import threshold_torch

    initial = torch.tensor([0, 3, 0, 1], dtype=torch.complex64)
    replacement = torch.tensor([0, 1, 0, 4], dtype=torch.complex64)
    monkeypatch.setenv(_GATE, "yes")

    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(initial)
        first = engine.threshold_and_cluster(0.5, 2)
        first_bytes = tuple(_result_bytes(result) for result in first)
        first_pointers = tuple(result.ptr for result in first)

        initial.copy_(replacement)
        second = engine.threshold_and_cluster(0.5, 2)

        assert tuple(_result_bytes(result) for result in first) == first_bytes
        assert all(
            first_pointer != second_result.ptr
            for first_pointer, second_result in zip(first_pointers, second)
        )
        np.testing.assert_array_equal(first[1].numpy(), [1])
        np.testing.assert_array_equal(first[0].numpy(), [3])
        np.testing.assert_array_equal(second[1].numpy(), [3])
        np.testing.assert_array_equal(second[0].numpy(), [4])


def test_trusted_threshold_result_fails_closed_for_other_route(monkeypatch):
    from pycbc import scheme
    from pycbc.events import threshold_torch

    monkeypatch.setenv(_GATE, "on")
    source = torch.tensor([0, 3, 0, 1], dtype=torch.float64)
    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(source)
        assert engine._trusted_native_result_scheme is None

        calls = []
        standard_wrapper = threshold_torch._array_from_tensor

        def record_standard_wrapper(tensor):
            calls.append(tensor)
            return standard_wrapper(tensor)

        with monkeypatch.context() as patch:
            patch.setattr(
                threshold_torch,
                "_array_from_tensor",
                record_standard_wrapper,
            )
            selected, indices = engine.threshold_and_cluster(0.5, 2)

        assert len(calls) == 2
        np.testing.assert_array_equal(indices.numpy(), [1])
        np.testing.assert_array_equal(selected.numpy(), [3])


def test_trusted_result_fails_closed_after_scheme_change(monkeypatch):
    import pycbc.scheme as pycbc_scheme
    from pycbc import scheme
    from pycbc.events import threshold_torch

    monkeypatch.setenv(_GATE, "1")
    source = torch.tensor([0, 3, 0, 1], dtype=torch.complex64)
    with scheme.TorchScheme("cpu"):
        engine = threshold_torch.TorchThresholdCluster(source)
        active_scheme = pycbc_scheme.mgr.state
        calls = []

        def record_fallback(tensor):
            calls.append(tensor)
            return tensor

        # Restore the real scheme before its context exits. This directly
        # exercises the private call site's identity guard without leaving the
        # process-global scheme manager in a modified state.
        with monkeypatch.context() as patch:
            patch.setattr(pycbc_scheme.mgr, "state", object())
            patch.setattr(
                threshold_torch, "_array_from_tensor", record_fallback
            )
            results = engine._native_cpu_threshold_and_cluster(
                np.float32(0.5), 2
            )

        assert pycbc_scheme.mgr.state is active_scheme
        assert len(calls) == 2
        assert results == tuple(calls)
