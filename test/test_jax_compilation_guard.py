"""The compilation guard detects real work and releases its listeners."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from jax_test_helpers import CompilationGuard  # noqa: E402


def test_guard_accepts_reuse_and_rejects_an_actual_new_compilation():
    first = jax.device_put(np.arange(13, dtype=np.float32))
    second = jax.device_put(np.arange(13, dtype=np.float32) + 1)
    changed = jax.device_put(np.arange(17, dtype=np.float32))
    traces = []

    @jax.jit
    def kernel(value):
        traces.append(value.shape)
        return value * np.float32(3) + np.float32(5)

    guard = CompilationGuard()
    try:
        kernel(first).block_until_ready()
        assert all(guard.snapshot()[name] > 0
                   for name in ("trace", "lower", "compile"))
        before = guard.snapshot()
        with guard.timed_guard():
            output = kernel(second)
            output.block_until_ready()
        assert guard.snapshot() == before
        np.testing.assert_array_equal(np.asarray(output),
                                      np.asarray(second) * 3 + 5)
        with pytest.raises(RuntimeError, match="tracing or compilation"):
            with guard.timed_guard():
                kernel(changed).block_until_ready()
        assert traces == [(13,), (17,)]
    finally:
        guard.close()


def test_guard_cleanup_releases_both_listeners_after_a_failure():
    guard = CompilationGuard()
    try:
        with pytest.raises(ValueError, match="input failed"):
            with guard.timed_guard():
                raise ValueError("input failed")
    finally:
        guard.close()
    before = guard.snapshot()
    jax.monitoring.record_event("/jax/compilation_cache/cache_hits")
    jax.monitoring.record_event_duration_secs(
        "/jax/core/compile/backend_compile_duration", 0.1)
    assert guard.snapshot() == before
    guard.close()
