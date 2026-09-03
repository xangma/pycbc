"""Exactness and fail-closed tests for the optional XAS phase-plan trace."""

import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pycbc.waveform._torch_jax import torch_context  # noqa: E402
from pycbc.waveform import imrphenomx_utils_torch as xutils  # noqa: E402
from pycbc.waveform import imrphenomxas_torch as xas  # noqa: E402


_TRACE_ENV = "PYCBC_IMRPHENOMXAS_PHASE_PLAN_TORCHSCRIPT_TRACE"
_CONTROL_ENV = {
    "PYCBC_IMRPHENOMX_PHASE_PLAN_BULK_COLLOCATION": "0",
    "PYCBC_IMRPHENOMX_PHASE_PLAN_CUDA_SOLVE_GRAPH": "0",
    "PYCBC_IMRPHENOMX_SCALAR_REGION_DISPATCH": "0",
    "PYCBC_IMRPHENOMX_EXACT_SCALAR_DERIVATIVES": "0",
    "PYCBC_IMRPHENOMXAS_SCALAR_DERIVATIVE_PLAN_CSE": "0",
    "PYCBC_IMRPHENOMXAS_INSPIRAL_PHASE_HOST_SCALARS": "0",
    "PYCBC_IMRPHENOMXAS_SCRIPTED_PHASE_ANSATZ_CPU": "0",
    "PYCBC_IMRPHENOMXAS_CUDA_GRAPH_PHASE_ANSATZ": "0",
}


requires_phase_plan_trace_runtime = pytest.mark.skipif(
    not xas._phase_plan_torchscript_trace_runtime_supported(
        torch.device("cpu")
    ),
    reason="phase-plan TorchScript tracing is unsupported by this runtime",
)


@pytest.fixture(autouse=True)
def _controlled_trace_environment(monkeypatch):
    for name, value in _CONTROL_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(_TRACE_ENV, raising=False)
    xas._clear_phase_plan_torchscript_trace_cache()
    yield
    xas._clear_phase_plan_torchscript_trace_cache()


def _phase_call(values=(35.0, 12.0, 0.4, -0.3)):
    theta = torch.tensor(values, dtype=torch.float64)
    coefficients = xutils.get_phenomx_phase_coeff_table(
        device=theta.device,
        dtype=theta.dtype,
    )
    chip = 0.2
    final_spin = 0.7
    with torch_context(theta):
        cutoff = xas._get_cutoff_fMs(
            *theta,
            chip=chip,
            final_spin=final_spin,
        )

    def call():
        with torch_context(theta):
            return xas._prepare_phase_plan(
                theta,
                coefficients,
                chip,
                final_spin=final_spin,
                _cutoff_fMs=cutoff,
            )

    return call, theta, coefficients, cutoff


def test_phase_plan_trace_gate_is_strict_and_off_by_default(monkeypatch):
    assert not xas._phase_plan_torchscript_trace_enabled()
    monkeypatch.setenv(_TRACE_ENV, "0")
    assert not xas._phase_plan_torchscript_trace_enabled()
    monkeypatch.setenv(_TRACE_ENV, "1")
    assert xas._phase_plan_torchscript_trace_enabled()
    monkeypatch.setenv(_TRACE_ENV, "maybe")
    with pytest.raises(ValueError):
        xas._phase_plan_torchscript_trace_enabled()


@requires_phase_plan_trace_runtime
def test_phase_plan_trace_runtime_contract_fails_closed(monkeypatch):
    cpu = torch.device("cpu")
    assert not xas._phase_plan_torchscript_trace_runtime_supported(torch.device("cuda"))
    with torch.no_grad():
        assert not xas._phase_plan_torchscript_trace_runtime_supported(cpu)
    with torch.inference_mode():
        assert not xas._phase_plan_torchscript_trace_runtime_supported(cpu)
    with torch.autograd.forward_ad.dual_level():
        assert not xas._phase_plan_torchscript_trace_runtime_supported(cpu)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert not xas._phase_plan_torchscript_trace_runtime_supported(cpu)

    for version, expected in (
        ("2.8.9", False),
        ("2.9.0", True),
        ("2.13.0", True),
        ("2.14.0", False),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(torch, "__version__", version)
            assert xas._phase_plan_torchscript_trace_runtime_supported(cpu) is expected
    with monkeypatch.context() as patch:
        patch.setattr(torch.compiler, "is_compiling", lambda: True)
        assert not xas._phase_plan_torchscript_trace_runtime_supported(cpu)
    with monkeypatch.context() as patch:
        patch.setattr(
            torch,
            "are_deterministic_algorithms_enabled",
            lambda: True,
        )
        assert not xas._phase_plan_torchscript_trace_runtime_supported(cpu)


def test_phase_plan_trace_python_contract_fails_closed(monkeypatch):
    cases = (
        ((3, 11, 9), "cpython", None, True),
        ((3, 12, 7), "cpython", None, True),
        ((3, 13, 9), "cpython", True, True),
        ((3, 13, 9), "cpython", False, False),
        ((3, 13, 9), "pypy", True, False),
        ((3, 14, 2), "cpython", False, False),
        ((3, 14, 6), "cpython", True, False),
    )
    for version, implementation, gil_enabled, expected in cases:
        fake_sys = SimpleNamespace(
            version_info=version,
            implementation=SimpleNamespace(name=implementation),
        )
        if gil_enabled is not None:
            fake_sys._is_gil_enabled = lambda value=gil_enabled: value
        with monkeypatch.context() as patch:
            patch.setattr(xas, "sys", fake_sys)
            assert xas._phase_plan_torchscript_trace_python_supported() is expected


def test_phase_plan_trace_key_includes_thread_topology(monkeypatch):
    _, theta, coefficients, cutoff = _phase_call()
    with torch_context(theta):
        fit_rows = xas._prepare_phase_fit_rows(theta, coefficients)
    first = xas._phase_plan_torchscript_trace_key(
        theta,
        coefficients,
        fit_rows,
        cutoff,
    )
    with monkeypatch.context() as patch:
        patch.setattr(torch, "get_num_threads", lambda: 3)
        patch.setattr(torch, "get_num_interop_threads", lambda: 2)
        second = xas._phase_plan_torchscript_trace_key(
            theta,
            coefficients,
            fit_rows,
            cutoff,
        )
    assert first != second
    assert second[4] == (3, 2)


def test_phase_plan_trace_resets_inherited_cache(monkeypatch):
    xas._PHASE_PLAN_TORCHSCRIPT_TRACE_CACHE["ready"] = object()
    monkeypatch.setattr(
        xas,
        "_PHASE_PLAN_TORCHSCRIPT_TRACE_PID",
        os.getpid() - 1,
    )
    assert xas._phase_plan_torchscript_trace_cache_state() == ((), ())
    assert xas._PHASE_PLAN_TORCHSCRIPT_TRACE_PID == os.getpid()


@requires_phase_plan_trace_runtime
def test_phase_plan_trace_cold_and_warm_are_byte_exact(monkeypatch):
    call, _, _, _ = _phase_call()
    monkeypatch.setenv(_TRACE_ENV, "0")
    reference = call()
    monkeypatch.setenv(_TRACE_ENV, "1")
    cold = call()
    warm = call()
    assert xas._phase_plan_torchscript_trace_cache_state()[1] == ()
    assert xas._phase_plan_torchscript_tree_raw_equal(reference, cold)
    assert xas._phase_plan_torchscript_tree_raw_equal(reference, warm)
    tensors = tuple(
        value
        for value in xas._phase_plan_torchscript_primary_values(warm)
        if type(value) is torch.Tensor
    )
    assert len(tensors) == 37
    assert all(value.dtype is torch.float64 for value in tensors)


@requires_phase_plan_trace_runtime
def test_phase_plan_trace_preserves_alias_and_mutation_semantics(monkeypatch):
    alias_of = torch._C._is_alias_of

    def alias_signature(plan):
        leaves = xas._phase_plan_tensor_leaves(plan)
        identity = tuple(
            (left, right)
            for left in range(len(leaves))
            for right in range(left + 1, len(leaves))
            if leaves[left] is leaves[right]
        )
        storage = tuple(
            (left, right)
            for left in range(len(leaves))
            for right in range(left + 1, len(leaves))
            if alias_of(leaves[left], leaves[right])
        )
        return identity, storage

    def exercise(enabled):
        call, _, _, _ = _phase_call()
        monkeypatch.setenv(_TRACE_ENV, str(int(enabled)))
        xas._clear_phase_plan_torchscript_trace_cache()
        call()  # Build the executor when enabled; otherwise an eager pre-call.
        first = call()
        second = call()
        first_values = tuple(
            value
            for value in xas._phase_plan_torchscript_primary_values(first)
            if type(value) is torch.Tensor
        )
        second_values = tuple(
            value
            for value in xas._phase_plan_torchscript_primary_values(second)
            if type(value) is torch.Tensor
        )
        cross_call_identity = tuple(
            index
            for index, (left, right) in enumerate(zip(first_values, second_values))
            if left is right
        )
        assert cross_call_identity == (22, 23, 29, 30)

        with torch.no_grad():
            first_values[22].add_(0.125)
        mutated_hex = first_values[22].item().hex()
        assert first_values[29].item().hex() == mutated_hex
        assert second_values[22].item().hex() == mutated_hex
        assert second_values[29].item().hex() == mutated_hex
        after = call()
        after_values = tuple(
            value
            for value in xas._phase_plan_torchscript_primary_values(after)
            if type(value) is torch.Tensor
        )
        assert after_values[22].item().hex() == mutated_hex
        assert after_values[29].item().hex() == mutated_hex
        return alias_signature(first), mutated_hex, after

    eager_aliases, eager_hex, eager_after = exercise(False)
    traced_aliases, traced_hex, traced_after = exercise(True)
    assert traced_aliases == eager_aliases
    assert traced_hex == eager_hex
    assert xas._phase_plan_torchscript_tree_raw_equal(eager_after, traced_after)
    assert xas._phase_plan_torchscript_tree_alias_equal(eager_after, traced_after)


@requires_phase_plan_trace_runtime
def test_phase_plan_trace_build_failure_is_sticky_eager_fallback(monkeypatch):
    call, _, _, _ = _phase_call()
    monkeypatch.setenv(_TRACE_ENV, "0")
    reference = call()
    builds = 0

    def fail(_inputs):
        nonlocal builds
        builds += 1
        raise RuntimeError("synthetic trace failure")

    monkeypatch.setattr(xas, "_build_phase_plan_torchscript_trace", fail)
    monkeypatch.setenv(_TRACE_ENV, "1")
    first = call()
    second = call()
    assert builds == 1
    assert tuple(map(len, xas._phase_plan_torchscript_trace_cache_state())) == (
        0,
        1,
    )
    assert xas._phase_plan_torchscript_tree_raw_equal(reference, first)
    assert xas._phase_plan_torchscript_tree_raw_equal(reference, second)
