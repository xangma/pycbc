"""Lazy JAX slices retain write-through storage and metadata contracts."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from pycbc import scheme  # noqa: E402
from pycbc.types import Array, FrequencySeries, TimeSeries  # noqa: E402
from pycbc.types.array_jax import JAXArrayData  # noqa: E402


def _devices():
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("cuda:0")
    except RuntimeError:
        pass
    return devices


@pytest.fixture(params=_devices())
def jax_context(request):
    with scheme.JAXScheme(request.param) as context:
        yield context


SLICES = [slice(None), slice(2, 8), slice(-8, -1), slice(None, None, -1),
          slice(9, 1, -2), slice(1, 11, 3), slice(3, 3),
          slice(-100, 100, 2), slice(100, -100, -3)]


@pytest.mark.parametrize("shape", [(12,), (12, 3), (12, 2, 3)])
@pytest.mark.parametrize("index", SLICES)
def test_slice_creation_and_metadata_never_read_device_data(
        jax_context, shape, index, monkeypatch):
    host = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    parent = JAXArrayData(jnp.asarray(host))
    device = parent.device

    def reject_array(data):
        raise AssertionError("slice construction/metadata dispatched indexing")

    monkeypatch.setattr(JAXArrayData, "array", property(reject_array))
    child = parent[index]
    nested = child[::-2]
    for view, expected in [(child, host[index]),
                           (nested, host[index][::-2])]:
        assert view._array is None
        assert view.shape == expected.shape
        assert view.ndim == expected.ndim
        assert view.size == expected.size
        assert view.nbytes == expected.nbytes
        assert view.dtype == expected.dtype
        assert len(view) == len(expected)
        assert view.device == device


@pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("index", SLICES)
def test_lazy_slice_reads_and_assignments_track_current_parent(
        jax_context, dtype, index):
    host = np.arange(12, dtype=dtype)
    parent = JAXArrayData(jnp.asarray(host))
    child = parent[index]
    sibling = parent[::2]
    np.testing.assert_array_equal(child.array, host[index])

    replacement = np.arange(host[index].size, dtype=dtype) + 100
    child.set_array(replacement)
    host[index] = replacement
    np.testing.assert_array_equal(parent.array, host)
    np.testing.assert_array_equal(sibling.array, host[::2])

    parent.set_array(jnp.asarray(host + 10))
    np.testing.assert_array_equal(child.array, (host + 10)[index])
    assert child.array.dtype == np.dtype(dtype)
    assert child._array is None


def test_nested_reverse_and_strided_views_write_through(jax_context):
    host = np.arange(16, dtype=np.complex64)
    parent = JAXArrayData(jnp.asarray(host))
    child = parent[1:15:2]
    nested = child[::-1][1:6:2]
    target = host[1:15:2][::-1][1:6:2]
    nested.set_array(jnp.asarray([31 + 2j, 32 + 3j, 33 + 4j],
                                 dtype=jnp.complex128))
    target[:] = [31 + 2j, 32 + 3j, 33 + 4j]
    np.testing.assert_array_equal(parent.array, host)
    nested[1:] = jnp.asarray([41 + 5j, 42 + 6j], jnp.complex128)
    target[1:] = [41 + 5j, 42 + 6j]
    np.testing.assert_array_equal(parent.array, host)
    assert nested.dtype == np.dtype(np.complex64)
    assert nested._array is None


def test_broadcast_assignment_fill_and_inplace_preserve_slice_storage(jax_context):
    host = np.arange(12, dtype=np.float32)
    parent = JAXArrayData(jnp.asarray(host))
    child = parent[2:10:2]
    child.set_array(jnp.asarray([50], jnp.float64))
    host[2:10:2] = 50
    np.testing.assert_array_equal(parent.array, host)
    child.fill(7)
    host[2:10:2] = 7
    child += 3
    host[2:10:2] += 3
    np.testing.assert_array_equal(parent.array, host)
    assert child.array.dtype == jnp.float32
    assert child._array is None


@pytest.mark.parametrize("index", SLICES)
def test_slice_assignment_broadcast_and_errors_match_numpy(jax_context, index):
    host = np.arange(12, dtype=np.float32)
    parent = Array(host)
    parent[index] = Array(np.asarray([23], dtype=np.float32))
    host[index] = 23
    np.testing.assert_array_equal(parent.numpy(), host)

    if host[index].size not in (0, 2):
        with pytest.raises(ValueError):
            parent[index] = Array(np.asarray([4, 5], dtype=np.float32))
        np.testing.assert_array_equal(parent.numpy(), host)


def test_jax_array_helpers_follow_array_contracts(jax_context):
    parent = Array(np.asarray([1, 2, 3, 4], dtype=np.float32))
    selected = parent.take([0, -1])
    np.testing.assert_array_equal(selected.numpy(), [1, 4])
    assert selected.dtype == parent.dtype
    with pytest.raises(IndexError):
        parent.take([len(parent)])

    view = parent[1:3]
    result = view.multiply_and_add(Array(np.asarray([5, 7], np.float32)), 2)
    np.testing.assert_array_equal(parent.numpy(), [1, 12, 17, 4])
    np.testing.assert_array_equal(result.numpy(), [12, 17])
    view[:] = 9
    np.testing.assert_array_equal(result.numpy(), [9, 9])

    complex_values = Array(np.asarray([1 + 1j, 3 + 4j, -3 - 4j], np.complex64))
    assert complex_values.abs_arg_max() == 1
    assert parent.abs_arg_max() == 1
    with pytest.raises(NotImplementedError, match="write-through dtype views"):
        parent.view(np.int32)


def test_copy_detaches_view_and_metadata_tracks_parent_replacement(
        jax_context):
    parent = JAXArrayData(jnp.arange(12, dtype=jnp.float32))
    child = parent[2:10:2]
    copied = child.copy()
    child.fill(90)
    np.testing.assert_array_equal(copied.array, [2, 4, 6, 8])
    assert copied.parent is None
    parent.array = jnp.arange(6, dtype=jnp.float32)
    assert child.shape == (2,)
    assert len(child) == 2
    np.testing.assert_array_equal(child.array, [2, 4])


def test_non_slice_indexing_and_invalid_slices_keep_legacy_path(jax_context):
    parent = JAXArrayData(jnp.arange(12, dtype=jnp.float32).reshape(4, 3))
    for index in [1, (slice(None), 1), jnp.asarray([3, 0])]:
        np.testing.assert_array_equal(parent[index].array, parent.array[index])
    assert parent[1][2] == 5
    for index, error_type in [(slice(None, None, 0), ValueError),
                              (slice(1.5, 4), TypeError)]:
        with pytest.raises(error_type):
            parent[index]
    scalar = JAXArrayData(jnp.asarray(1, dtype=jnp.float32))
    assert scalar.shape == () and scalar.ndim == 0 and scalar.size == 1
    with pytest.raises(TypeError):
        len(scalar)
    with pytest.raises(IndexError):
        scalar[:]


@pytest.mark.parametrize("series_type", [Array, FrequencySeries, TimeSeries])
def test_pycbc_series_metadata_and_parent_updates_without_materializing_views(
        jax_context, series_type, monkeypatch):
    options = {FrequencySeries: dict(delta_f=.25, epoch=1234),
               TimeSeries: dict(delta_t=.125, epoch=1234)}.get(series_type, {})
    parent = series_type(np.arange(12, dtype=np.float32), **options)
    original = JAXArrayData.array

    def reject_view(data):
        if data.parent is not None:
            raise AssertionError("PyCBC metadata resolved a device view")
        return original.fget(data)

    with monkeypatch.context() as patch:
        patch.setattr(JAXArrayData, "array",
                      property(reject_view, original.fset))
        child = parent[2:10:2]
        assert len(child) == 4 and child.shape == (4,)
        assert child.dtype == np.dtype(np.float32)
        assert child._data._array is None
        if series_type is FrequencySeries:
            assert child.delta_f == .5
            assert child.epoch == parent.epoch
        elif series_type is TimeSeries:
            assert child.delta_t == .25
            assert child.start_time == parent.start_time + .25
    parent[2:10:2] = Array(np.full(4, 9, np.float32))
    np.testing.assert_array_equal(child.numpy(), np.full(4, 9, np.float32))


def test_static_lazy_slices_work_inside_jit(jax_context):
    def sliced(values):
        view = JAXArrayData(values)[::-2][1:]
        assert view.shape == (4,) and view._array is None
        return view.array

    values = jnp.arange(10, dtype=jnp.float32)
    np.testing.assert_array_equal(jax.jit(sliced)(values), values[::-2][1:])


@pytest.mark.parametrize("x64", [False, True])
def test_lazy_metadata_preserves_canonical_dtype_with_x64_on_or_off(
        jax_context, monkeypatch, x64):
    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        from jax.experimental import enable_x64
    monkeypatch.setenv("PYCBC_JAX_ENABLE_X64", str(int(x64)))
    with enable_x64(x64):
        raw = jnp.asarray(np.arange(12, dtype=np.complex128))
        parent = JAXArrayData(raw)
        child = parent[1:11:3]
        assert child.dtype == np.dtype(raw.dtype)
        assert child.nbytes == 4 * raw.dtype.itemsize
        assert child.device == raw.device
        np.testing.assert_array_equal(child.array, raw[1:11:3])
        assert child._array is None


def test_sharded_slice_retains_legacy_device_metadata(jax_context):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    devices = jax.devices(jax_context.jax_device.platform)
    if len(devices) < 2:
        pytest.skip("multiple JAX devices unavailable")
    mesh = Mesh(np.asarray(devices[:2]), ("x",))
    sharding = NamedSharding(mesh, PartitionSpec("x"))
    raw = jax.device_put(jnp.arange(12, dtype=jnp.float32), sharding)
    parent = JAXArrayData(raw)
    child = parent[2:6]
    assert child._array is not None
    assert child.device == raw[2:6].device
    np.testing.assert_array_equal(child.array, raw[2:6])
