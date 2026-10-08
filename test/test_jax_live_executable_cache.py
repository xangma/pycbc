# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under the GNU General Public License, version 3 or later.

"""Regression coverage for retained Live executable and resident bin caches."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")





from pycbc.vetoes import chisq_jax





@pytest.fixture(autouse=True)
def isolated_retained_handles(monkeypatch):
    monkeypatch.setattr(chisq_jax, "_LIVE_EXECUTABLE_CACHE", OrderedDict())


class _CountingFunction:
    __name__ = "counting_live_kernel"

    def __init__(self):
        self.lowerings = []

    def lower(self, *args, **kwargs):
        self.lowerings.append((args, kwargs))
        executable = object()
        return SimpleNamespace(compile=lambda: executable)


def test_handle_survives_search_reconstruction_without_caching_values():
    device = jax.devices("cpu")[0]
    function = _CountingFunction()
    args = (jax.ShapeDtypeStruct((8,), np.float32),)
    first = chisq_jax._live_chisq_executable(function, args, {}, {}, device)
    rebuilt = {}
    second = chisq_jax._live_chisq_executable(function, args, {}, rebuilt, device)
    assert first is second
    assert len(function.lowerings) == 1
    assert tuple(rebuilt.values()) == (first,)


def test_rebuilt_search_executes_current_scientific_values():
    device = jax.devices("cpu")[0]

    @jax.jit
    def scientific(values):
        return values * values + 3

    first_values = jax.device_put(np.arange(8, dtype=np.float32), device)
    later_values = first_values + 7
    first = chisq_jax._live_chisq_executable(
        scientific, (first_values,), {}, {}, device)
    later = chisq_jax._live_chisq_executable(
        scientific, (later_values,), {}, {}, device)
    assert first is later
    np.testing.assert_array_equal(np.asarray(later(later_values)),
                                  np.arange(7, 15, dtype=np.float32) ** 2 + 3)


def test_handle_key_separates_shapes_dtypes_static_values_and_functions():
    device = jax.devices("cpu")[0]
    function = _CountingFunction()
    cache = {}
    cases = [((8,), np.float32, 1), ((9,), np.float32, 1),
             ((8,), np.float64, 1), ((8,), np.float32, 2)]
    handles = [chisq_jax._live_chisq_executable(
        function, (jax.ShapeDtypeStruct(shape, dtype),),
        {"count": count}, cache, device) for shape, dtype, count in cases]
    assert len(set(handles)) == 4
    other = chisq_jax._live_chisq_executable(
        _CountingFunction(), (jax.ShapeDtypeStruct((8,), np.float32),),
        {"count": 1}, cache, device)
    assert other not in handles


def test_handle_key_preserves_weak_type_sharding_and_precision_config():
    device = jax.devices("cpu")[0]
    function = _CountingFunction()
    single = jax.sharding.SingleDeviceSharding(device)
    named = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.asarray([device]), ("worker",)),
        jax.sharding.PartitionSpec())
    handles = []
    for sharding, weak in ((single, False), (single, True), (named, False)):
        arg = jax.ShapeDtypeStruct((), np.float32, sharding=sharding,
                                  weak_type=weak)
        handles.append(chisq_jax._live_chisq_executable(
            function, (arg,), {}, {}, device))
        abstract = function.lowerings[-1][0][0]
        assert abstract.sharding == sharding
        assert abstract.weak_type is weak
    assert len(set(handles)) == 3
    arg = jax.ShapeDtypeStruct((), np.float32, sharding=single)
    with jax.default_matmul_precision("highest"):
        changed = chisq_jax._live_chisq_executable(
            function, (arg,), {}, {}, device)
    assert changed not in handles


def test_handle_cache_separates_devices():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("Needs two CPU devices or XLA_FLAGS=--xla_force_host_platform_device_count=2")
    function = _CountingFunction()
    args = (jax.ShapeDtypeStruct((8,), np.float32),)
    handles = [chisq_jax._live_chisq_executable(
        function, args, {}, {}, device) for device in devices[:2]]
    assert handles[0] is not handles[1]


def test_handle_cache_separates_x64_modes():
    device = jax.devices("cpu")[0]
    function = _CountingFunction()
    args = (jax.ShapeDtypeStruct((8,), np.float32),)
    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        from jax.experimental import enable_x64
    with enable_x64(False):
        narrow = chisq_jax._live_chisq_executable(function, args, {}, {}, device)
    with enable_x64(True):
        wide = chisq_jax._live_chisq_executable(function, args, {}, {}, device)
    assert narrow is not wide


def test_local_and_process_handle_caches_are_bounded_lru(monkeypatch):
    monkeypatch.setattr(chisq_jax, "_LIVE_EXECUTABLE_CACHE_LIMIT", 2)
    device = jax.devices("cpu")[0]
    function = _CountingFunction()
    cache = {}

    def handle(size):
        return chisq_jax._live_chisq_executable(
            function, (jax.ShapeDtypeStruct((size,), np.float32),), {}, cache,
            device)

    first, second = handle(1), handle(2)
    assert handle(1) is first
    third = handle(3)
    assert len(cache) == len(chisq_jax._LIVE_EXECUTABLE_CACHE) == 2
    assert set(cache.values()) == {first, third}
    assert handle(2) is not second




