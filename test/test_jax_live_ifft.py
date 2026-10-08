# Copyright (C) 2026 The PyCBC Collaboration
"""Full-buffer Live IFFT parity, ownership and generic fallback tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from pycbc import scheme
from pycbc.fft import IFFT
from pycbc.types import Array

jax = pytest.importorskip("jax")
from pycbc.filter import matchedfilter_jax as backend  # noqa: E402


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param == "cuda:0":
        try:
            available = jax.devices("gpu")
        except RuntimeError:
            available = []
        if not available:
            pytest.skip("CUDA JAX device unavailable")
    return request.param


@pytest.mark.parametrize("count,size", [(1, 12), (3, 16), (2, 30), (3, 49152)])
def test_flat_live_ifft_preserves_inputs_and_matches_generic(device, count, size):
    """Full output bytes and input aliases survive successive block updates."""
    with scheme.JAXScheme(device):
        rng = np.random.default_rng(8802)
        source = Array(np.zeros(count * size, np.complex64))
        reference, output = (Array(np.zeros(count * size, np.complex64))
                             for _ in range(2))
        plan = IFFT(source, output, nbatch=count, size=size)
        generic = IFFT(source, reference, nbatch=count, size=size)
        backend._bind_live_ifft_workspace_jax(plan)
        assert (plan._jax_live_ifft_workspace is not None) == (device != "cpu")
        alias = output[size // 2:size + 2]
        for scale in (np.complex64(1), np.complex64(0.25 - 0.5j)):
            # Nonzero negative-frequency halves detect destructive FFT reuse.
            values = ((rng.normal(size=count * size)
                       + 1j * rng.normal(size=count * size)) * scale
                      ).astype(np.complex64)
            source._data.set_array(jax.numpy.asarray(values))
            retained_input = source._data.array
            retained_output = output._data.array
            previous = np.asarray(retained_output).copy()
            generic.execute()
            backend._live_ifft_execute_jax(plan)
            flat = backend._live_ifft_flat_jax(source._data.array, count, size)
            np.testing.assert_array_equal(np.asarray(output), np.asarray(reference))
            np.testing.assert_array_equal(np.asarray(flat), np.asarray(reference))
            np.testing.assert_array_equal(np.asarray(source), values)
            np.testing.assert_array_equal(np.asarray(retained_input), values)
            np.testing.assert_array_equal(np.asarray(retained_output), previous)
            np.testing.assert_array_equal(np.asarray(alias),
                                          np.asarray(reference)[size // 2:size + 2])


@pytest.mark.parametrize("change", [
    "invec", "outvec", "source", "target", "shape", "dtype", "parent",
    "slice", "geometry", "compiled", "inplace", "shared_array",
])
def test_live_ifft_drops_changed_workspace_and_calls_original(device, change):
    """Unqualified mutable plans retain their original execution semantics."""
    if device == "cpu":
        pytest.skip("Workspace specialization is CUDA only")
    with scheme.JAXScheme(device):
        source = Array(np.zeros(32, np.complex64))
        target = Array(np.zeros(32, np.complex64))
        plan = IFFT(source, target, nbatch=2, size=16)
        backend._bind_live_ifft_workspace_jax(plan)
        assert plan._jax_live_ifft_workspace is not None
        if change == "invec":
            plan.invec = Array(source, copy=False)
        elif change == "outvec":
            plan.outvec = Array(target, copy=False)
        elif change == "source":
            source._data = Array(np.ones(32, np.complex64))._data
        elif change == "target":
            target._data = Array(np.ones(32, np.complex64))._data
        elif change == "shape":
            # Deliberately replace the immutable root; assignment preserves size.
            source._data.array = jax.numpy.ones(31, jax.numpy.complex64)
        elif change == "dtype":
            source._data.array = source._data.array.astype(jax.numpy.complex128)
        elif change == "parent":
            target._data.parent = source._data
        elif change == "slice":
            target._data.slice_info = slice(0, 32)
        elif change == "geometry":
            plan.size = 8
        elif change == "compiled":
            plan._compiled = lambda value: value
        elif change == "inplace":
            plan.inplace = True
        elif change == "shared_array":
            target._data.array = source._data.array
        calls = []
        plan.execute = lambda: calls.append(True)
        backend._live_ifft_execute_jax(plan)
        assert calls == [True]
        assert plan._jax_live_ifft_workspace is None


def test_live_ifft_accepts_existing_fallback_plan():
    calls = []
    plan = SimpleNamespace(execute=lambda: calls.append(True))
    backend._live_ifft_execute_jax(plan)
    assert calls == [True]
