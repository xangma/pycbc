# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under the GNU General Public License, version 3 or later.

"""Terminal collection byte, placement, and dispatch contracts."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from pycbc.events import live_collect_jax as collect  # noqa: E402
from pycbc.types.array_jax import _ensure_x64  # noqa: E402


def _devices():
    devices = [jax.devices("cpu")[0]]
    try:
        devices.extend(jax.devices("gpu")[:1])
    except RuntimeError:
        pass
    return devices


def _assert_contract(actual, expected):
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    assert actual_tree == expected_tree
    for left, right in zip(actual_leaves, expected_leaves):
        assert type(left) is type(right)
        if isinstance(right, np.ndarray):
            assert left.shape == right.shape and left.dtype == right.dtype
            assert left.tobytes() == right.tobytes()
            assert left.flags.writeable == right.flags.writeable
        else:
            assert left is right or left == right


def _tree(device):
    real = np.asarray(
        [0x7FC01234, 0x80000000, 0x7FA01234, 0x3F800000], np.uint32
    ).view(np.float32)
    wide = np.asarray(
        [0x7FF8000000001234, 0x8000000000000000], np.uint64
    ).view(np.float64)
    values = [
        real,
        wide,
        real.view(np.complex64),
        wide.view(np.complex128),
        np.asarray([True, False]),
        np.asarray([0, 2**32 - 1], np.uint32),
    ]
    columns = []
    for value in values:
        columns.append(
            (
                jax.device_put(value, device),
                jax.device_put(value[:1].reshape(()), device),
                jax.device_put(np.empty((0, 3), value.dtype), device),
            )
        )
    return {
        "columns": columns,
        "native": [np.str_("H1"), np.int64(7), np.arange(3), "L1", 3.0, None],
        "opaque": object(),
    }


@pytest.mark.parametrize("device", _devices())
def test_terminal_collection_matches_device_get_contract_and_bytes(device):
    _ensure_x64()
    tree = _tree(device)
    expected = jax.device_get(tree)
    actual = collect.collect_live_arrays(tree)
    _assert_contract(actual, expected)
    # Native arrays retain device_get's identity; packed views retain owners.
    assert actual["native"][2] is tree["native"][2]
    saved = actual["columns"][0][0].tobytes()
    del tree
    assert actual["columns"][0][0].tobytes() == saved


@pytest.mark.parametrize("device", _devices())
def test_warm_collection_uses_one_transfer_and_compiled_packing(
    monkeypatch, device
):
    _ensure_x64()
    tree = {
        "columns": [
            jax.device_put(np.arange(11, dtype=np.float32) + index, device)
            for index in range(24)
        ],
        "native": np.str_("H1"),
    }
    expected = jax.device_get(tree)
    collect.collect_live_arrays(tree)
    cache_size = collect._pack_live_arrays._cache_size()
    original_get = jax.device_get
    array_type = type(tree["columns"][0])
    original_array = array_type.__array__
    collecting = False
    calls = []

    def reject(*args, **kwargs):
        raise AssertionError(
            "eager array operation or readback before terminal collection"
        )

    def guarded_array(self, *args, **kwargs):
        if not collecting:
            reject()
        return original_array(self, *args, **kwargs)

    def device_get(tree):
        nonlocal collecting
        native, packed = tree
        assert len(packed) == 1 and packed[0].shape == (24 * 11,)
        assert not any(isinstance(value, jax.Array) for value in native)
        calls.append(tree)
        collecting = True
        try:
            return original_get(tree)
        finally:
            collecting = False

    with monkeypatch.context() as patch:
        patch.setattr(jax, "device_get", device_get)
        patch.setattr(jnp, "ravel", reject)
        patch.setattr(jnp, "concatenate", reject)
        patch.setattr(array_type, "__array__", guarded_array)
        for name in ("__float__", "__int__", "__bool__"):
            patch.setattr(array_type, name, reject)
        actual = collect.collect_live_arrays(tree)
    assert len(calls) == 1
    assert collect._pack_live_arrays._cache_size() == cache_size
    _assert_contract(actual, expected)


def test_packing_never_crosses_device_placement(monkeypatch):
    devices = list(jax.devices("cpu"))[:2]
    if len(devices) < 2:
        devices = _devices()
    if len(devices) < 2:
        pytest.skip("two JAX devices unavailable")
    arrays = [
        jax.device_put(np.arange(5, dtype=np.int32), device)
        for device in devices
        for _ in range(2)
    ]
    expected = jax.device_get(arrays)
    original = collect._pack_live_arrays
    groups = []

    def pack(*values):
        placements = {next(iter(value.devices())) for value in values}
        assert len(placements) == 1
        groups.append(placements)
        result = original(*values)
        assert result.devices() == placements
        return result

    monkeypatch.setattr(collect, "_pack_live_arrays", pack)
    with jax.default_device(devices[-1]):
        actual = collect.collect_live_arrays(arrays)
    assert groups == [{device} for device in devices]
    _assert_contract(actual, expected)


def test_unsupported_dtype_and_tracer_follow_device_get(monkeypatch):
    values = [jnp.arange(5, dtype=jnp.bfloat16), jnp.zeros(0, jnp.bfloat16)]
    assert all(collect._packing_key(value) is None for value in values)

    def reject(*args, **kwargs):
        raise AssertionError("unsupported dtype was packed")

    monkeypatch.setattr(collect, "_pack_live_arrays", reject)
    _assert_contract(
        collect.collect_live_arrays(values), jax.device_get(values)
    )
    function = jax.jit(
        lambda value: collect.collect_live_arrays({"value": value})
    )
    result = function(jnp.arange(4, dtype=jnp.int32))
    np.testing.assert_array_equal(
        result["value"], np.arange(4, dtype=np.int32)
    )


def test_multidevice_arrays_keep_original_collection(monkeypatch):
    devices = jax.devices("cpu")[:2]
    if len(devices) < 2:
        pytest.skip("two CPU devices unavailable")
    mesh = jax.sharding.Mesh(np.asarray(devices), ("rows",))
    placement = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("rows")
    )
    values = [
        jax.device_put(np.arange(8, dtype=np.int32), placement)
        for _ in range(2)
    ]
    assert all(collect._packing_key(value) is None for value in values)

    def reject(*args, **kwargs):
        raise AssertionError("multidevice array was packed")

    monkeypatch.setattr(collect, "_pack_live_arrays", reject)
    _assert_contract(
        collect.collect_live_arrays(values), jax.device_get(values)
    )


def test_extended_dtype_keeps_native_collection(monkeypatch):
    key = jax.random.key(17)
    assert collect._packing_key(key) is None

    def reject(*args, **kwargs):
        raise AssertionError("extended dtype was packed")

    monkeypatch.setattr(collect, "_pack_live_arrays", reject)
    actual = collect.collect_live_arrays({"key": key})["key"]
    expected = jax.device_get(key)
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(
        jax.random.key_data(actual), jax.random.key_data(expected)
    )
