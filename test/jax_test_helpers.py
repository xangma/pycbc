"""Test-only checks for reuse of warmed JAX executables."""

from contextlib import contextmanager
from threading import Lock

import jax


_EVENTS = {
    "/jax/compilation_cache/compile_requests_use_cache": "cache_request",
    "/jax/compilation_cache/cache_hits": "cache_hit",
    "/jax/compilation_cache/cache_misses": "cache_miss",
}
_DURATIONS = {
    "/jax/core/compile/jaxpr_trace_duration": "trace",
    "/jax/core/compile/jaxpr_to_mlir_module_duration": "lower",
    "/jax/core/compile/backend_compile_duration": "compile",
}
COUNTERS = tuple(_EVENTS.values()) + tuple(_DURATIONS.values())


class CompilationGuard:
    ("Observe public monitoring events "
     "without replacing compiler functions.")

    def __init__(self):
        self.counts = dict.fromkeys(COUNTERS, 0)
        self.lock = Lock()
        self.closed = False
        jax.monitoring.register_event_listener(self.event)
        jax.monitoring.register_event_duration_secs_listener(self.duration)

    def event(self, name, **metadata):
        counter = _EVENTS.get(name)
        if counter is not None:
            with self.lock:
                self.counts[counter] += 1

    def duration(self, name, seconds, **metadata):
        counter = _DURATIONS.get(name)
        if counter is not None:
            with self.lock:
                self.counts[counter] += 1

    def snapshot(self):
        with self.lock:
            return dict(self.counts)

    @contextmanager
    def timed_guard(self):
        before = self.snapshot()
        yield
        if self.snapshot() != before:
            raise RuntimeError(
                "JAX tracing or compilation occurred after warmup")

    def close(self):
        if not self.closed:
            jax.monitoring.unregister_event_listener(self.event)
            jax.monitoring.unregister_event_duration_listener(self.duration)
            self.closed = True
