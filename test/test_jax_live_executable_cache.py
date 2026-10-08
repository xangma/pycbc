# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under the GNU General Public License, version 3 or later.

"""Regression coverage for retained Live executable and resident bin caches."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pycbc import scheme
from pycbc.types import FrequencySeries
from pycbc.types.array_jax import _ensure_x64, to_jax
from pycbc.vetoes import chisq_jax
from pycbc.vetoes.chisq import SingleDetPowerChisq


def _devices():
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("cuda:0")
    except RuntimeError:
        pass
    return devices


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


@pytest.mark.parametrize("analytic", [False, True])
@pytest.mark.parametrize("dev_name", _devices())
def test_resident_bins_match_scalar_without_readback(monkeypatch, analytic, dev_name):
    _ensure_x64()
    with scheme.JAXScheme(dev_name):
        rng = np.random.default_rng(6729)
        values = (rng.normal(size=(3, 129))
                  + 1j * rng.normal(size=(3, 129))).astype(np.complex64)
        templates = []
        for index, value in enumerate(values):
            template = FrequencySeries(value, delta_f=0.25)
            template.f_lower = 2.0 if index < 2 else 3.0
            template.params = SimpleNamespace()
            template.approximant = "test"
            template.end_idx = 128
            template.cout = jnp.empty(256, dtype=jnp.complex64)
            templates.append(template)
        source = jnp.asarray(values)
        psd = FrequencySeries(np.linspace(1, 2, 129, dtype=np.float32),
                              delta_f=0.25)
        power = SingleDetPowerChisq("4")
        if analytic:
            cumulative = jnp.cumsum(jnp.arange(129, dtype=jnp.float32))
            psd.sigmasq_vec = {"test": cumulative}
            from pycbc.vetoes.chisq import power_chisq_bins_from_sigmasq_series
            oracle = [power_chisq_bins_from_sigmasq_series(
                np.asarray(cumulative), 4, int(t.f_lower / psd.delta_f), 128)
                for t in templates]
        else:
            oracle = [np.asarray(chisq_jax.power_chisq_bins_jax(
                t, 4, psd, t.f_lower)) for t in templates]

        def reject(*args, **kwargs):
            raise AssertionError("resident bin construction read back device data")

        with monkeypatch.context() as patch:
            patch.setattr(jax, "device_get", reject)
            groups = chisq_jax.resident_power_chisq_bins_jax(
                power, templates, psd, source)
            cached = chisq_jax.resident_power_chisq_bins_jax(
                power, templates, psd, source)
        assert cached is groups
        assert len(groups) == 2
        for positions, edges, base_k, n_time in groups:
            assert not positions.flags.writeable
            assert isinstance(edges, jax.Array)
            assert n_time == 256
            assert base_k == int(templates[positions[0]].f_lower / 0.25)
            np.testing.assert_array_equal(np.asarray(edges),
                                          np.asarray(oracle)[positions])
        assert not any(hasattr(t, "_bin_cache") for t in templates)

        # The wrapper remains the same while its immutable scientific values
        # change. A numerical cache hit would silently retain obsolete bins.
        psd._data.set_array(to_jax(psd) * jnp.linspace(1, 6, 129))
        changed = chisq_jax.resident_power_chisq_bins_jax(
            power, templates, psd, source)
        assert changed is not groups
        templates[0].f_lower = 4.0
        assert chisq_jax.resident_power_chisq_bins_jax(
            power, templates, psd, source) is not changed
        replaced = chisq_jax.resident_power_chisq_bins_jax(
            power, templates, psd, source + jnp.asarray(0.25, source.dtype))
        assert replaced is not changed
        for offset in range(8):
            new_psd = FrequencySeries(
                np.linspace(1, 2 + offset, 129, dtype=np.float32), delta_f=0.25)
            chisq_jax.resident_power_chisq_bins_jax(
                power, templates, new_psd, source)
        assert all(len(history) <= 4 for _, history
                   in power._jax_resident_bin_groups.values())


def test_resident_bins_retain_each_duration_group_and_release_dead_banks():
    import gc
    import weakref

    _ensure_x64()
    with scheme.JAXScheme("cpu"):
        template = FrequencySeries(np.ones(17, np.complex64), delta_f=0.25)
        template.params = SimpleNamespace()
        template.f_lower = 0.5
        template.cout = jnp.zeros(32, jnp.complex64)
        psds = [FrequencySeries(np.ones(17, np.float32) * (1 + index),
                                delta_f=0.25) for index in range(2)]
        matrices = [jnp.ones((1, 17), jnp.complex64) * (1 + index)
                    for index in range(6)]
        power = SingleDetPowerChisq("4")
        plans = [(matrix, psd, chisq_jax.resident_power_chisq_bins_jax(
            power, [template], psd, matrix))
            for matrix in matrices for psd in psds]
        for matrix, psd, plan in plans:
            assert chisq_jax.resident_power_chisq_bins_jax(
                power, [template], psd, matrix) is plan
        assert len(power._jax_resident_bin_groups) == 6
        dead_id, reference = id(matrices[0]), weakref.ref(matrices[0])
        plans.clear()
        del matrices[0]
        gc.collect()
        assert reference() is None
        chisq_jax.resident_power_chisq_bins_jax(
            power, [template], psds[0], matrices[0])
        assert dead_id not in power._jax_resident_bin_groups
