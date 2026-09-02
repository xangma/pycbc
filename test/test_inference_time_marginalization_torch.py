import sys
import types
from types import SimpleNamespace

import lal
import numpy as np
import pytest


try:
    import lal.utils  # noqa: F401
except ModuleNotFoundError:
    lal_utils = types.ModuleType("lal.utils")
    sys.modules["lal.utils"] = lal_utils
    lal.utils = lal_utils

from pycbc.inference.models import tools as marginalization_tools


@pytest.fixture(params=("cpu", "cuda", "mps"))
def torch_device(request):
    torch = pytest.importorskip("torch")
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    return torch.device(device)


def _legacy_sky_indices(weights, offsets, size):
    """Reproduce the per-detector host-index path being optimized."""
    indices = []
    selected = []
    for weight, offset in zip(weights, offsets):
        index = marginalization_tools.draw_sample(weight, size=size)
        selected.append(
            marginalization_tools._selected_values(
                weight, index, host=False
            )
        )
        indices.append(index + offset)
    reference = indices[0]
    relative = [reference - index for index in indices[1:]]
    return reference, relative, selected


@pytest.mark.parametrize("seed", (7, 8127, 49031))
def test_same_device_sky_indices_batch_one_host_transfer(
        torch_device, monkeypatch, seed):
    torch = pytest.importorskip("torch")
    dtype = torch.float32 if torch_device.type == "mps" else torch.float64
    generated_weights = np.random.default_rng(seed).normal(size=(3, 11))
    weights = [
        torch.tensor(
            values, device=torch_device, dtype=dtype,
            requires_grad=True,
        )
        for values in generated_weights
    ]
    offsets = [0, 2, -1]
    sample_count = 47

    np.random.seed(seed)
    expected = _legacy_sky_indices(
        [weight.detach() for weight in weights], offsets, sample_count
    )
    expected_next_random = np.random.random()

    transfers = []
    real_transfer = marginalization_tools._device_index_matrix_to_host

    def checked_transfer(indices):
        transfers.append(indices)
        assert len(indices) == len(weights)
        assert all(index.device.type == torch_device.type for index in indices)
        return real_transfer(indices)

    gathered_indices = []
    real_selected = marginalization_tools._selected_values

    def checked_selected(values, indices, *, host=True):
        if marginalization_tools._torch_tensor(values) is not None:
            assert isinstance(indices, torch.Tensor)
            assert indices.device.type == torch_device.type
            gathered_indices.append(indices)
        return real_selected(values, indices, host=host)

    monkeypatch.setattr(
        marginalization_tools,
        "_device_index_matrix_to_host",
        checked_transfer,
    )
    monkeypatch.setattr(
        marginalization_tools, "_selected_values", checked_selected
    )

    np.random.seed(seed)
    actual = marginalization_tools._draw_sky_time_indices(
        weights, offsets, sample_count
    )
    actual_next_random = np.random.random()

    assert len(transfers) == 1
    assert len(gathered_indices) == len(weights)
    np.testing.assert_array_equal(actual[0], expected[0])
    for actual_delay, expected_delay in zip(actual[1], expected[1]):
        np.testing.assert_array_equal(actual_delay, expected_delay)
    for actual_weight, expected_weight in zip(actual[2], expected[2]):
        torch.testing.assert_close(actual_weight, expected_weight)
        assert actual_weight.device.type == torch_device.type
    assert actual_next_random == expected_next_random

    sum(weight.sum() for weight in actual[2]).backward()
    assert all(weight.grad is not None for weight in weights)
    assert all(
        weight.grad.device.type == torch_device.type for weight in weights
    )


def test_mixed_backend_sky_indices_keep_legacy_path(monkeypatch):
    torch = pytest.importorskip("torch")
    weights = [
        torch.tensor([0.0, 2.0, 1.0], dtype=torch.float64),
        np.array([1.5, -0.5, 0.25]),
    ]
    offsets = [0, 3]

    np.random.seed(191)
    expected = _legacy_sky_indices(weights, offsets, 31)
    expected_next_random = np.random.random()

    def unexpected_bulk_transfer(_indices):
        raise AssertionError("mixed inputs must use the legacy host path")

    torch_index_types = []
    real_selected = marginalization_tools._selected_values

    def checked_selected(values, indices, *, host=True):
        if marginalization_tools._torch_tensor(values) is not None:
            torch_index_types.append(type(indices))
            assert isinstance(indices, np.ndarray)
        return real_selected(values, indices, host=host)

    monkeypatch.setattr(
        marginalization_tools,
        "_device_index_matrix_to_host",
        unexpected_bulk_transfer,
    )
    monkeypatch.setattr(
        marginalization_tools, "_selected_values", checked_selected
    )

    np.random.seed(191)
    actual = marginalization_tools._draw_sky_time_indices(
        weights, offsets, 31
    )
    actual_next_random = np.random.random()

    assert torch_index_types == [np.ndarray]
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1][0], expected[1][0])
    torch.testing.assert_close(actual[2][0], expected[2][0])
    np.testing.assert_array_equal(actual[2][1], expected[2][1])
    assert actual_next_random == expected_next_random


class _NumpyValues:
    def __init__(self, values):
        self._values = values

    def numpy(self):
        return self._values


class _FakeSnr:
    def __init__(self, values, start_time, delta_t=0.25):
        self.values = values
        self.start_time = start_time
        self.delta_t = delta_t
        self.end_time = start_time + len(values) * delta_t

    def squared_norm(self):
        if marginalization_tools._torch_tensor(self.values) is not None:
            return self.values
        return _NumpyValues(self.values)

    def time_slice(self, _start, _end, mode=None):
        assert mode == "nearest"
        return self


def _sky_model(sample_count, weights, *, empty_map=False):
    start = 1_400_000_000.1234567
    ra = np.array([0.4, 1.2])
    dec = np.array([-0.3, 0.7])
    fp = {
        "H1": np.array([0.2, 0.5]),
        "L1": np.array([-0.1, 0.4]),
    }
    fc = {
        "H1": np.array([0.6, 0.8]),
        "L1": np.array([0.3, -0.2]),
    }
    dtc = {
        "H1": np.array([0.01, 0.02]),
        "L1": np.array([-0.015, 0.025]),
    }
    dmap = {} if empty_map else {(0,): [0, 1]}
    bin_prior = {} if empty_map else {(0,): 1.0}
    tinfo = (
        dmap,
        start + 0.5,
        start + 1.0,
        fp,
        fc,
        ra,
        dec,
        dtc,
        bin_prior,
    )
    return SimpleNamespace(
        tinfo={"H1L1": tinfo},
        vsamples=sample_count,
        marginalize_vector_params={},
        marginalize_vector_weights=weights,
        _current_params={},
    )


def _sky_snrs(backend, device=None):
    torch = pytest.importorskip("torch")
    start = 1_400_000_000.1234567
    h1 = np.array([0.0, 0.0, 2000.0, 0.0, 0.0, 0.0])
    l1 = np.array([0.0, 2000.0, 0.0, 0.0, 0.0, 0.0])
    leaves = []
    if backend == "torch":
        dtype = torch.float32 if device.type == "mps" else torch.float64
        h1 = torch.tensor(
            h1, device=device, dtype=dtype, requires_grad=True
        )
        l1 = torch.tensor(
            l1, device=device, dtype=dtype, requires_grad=True
        )
        leaves = [h1, l1]
    return {
        "H1": _FakeSnr(h1, start),
        "L1": _FakeSnr(l1, start + 0.25),
    }, leaves


def _draw_full_sky(backend, sample_count, seed, device=None, empty=False):
    torch = pytest.importorskip("torch")
    snrs, leaves = _sky_snrs(backend, device=device)
    if backend == "torch":
        dtype = torch.float32 if device.type == "mps" else torch.float64
        initial_weights = torch.zeros(
            sample_count, device=device, dtype=dtype
        )
    else:
        initial_weights = np.zeros(sample_count)
    model = _sky_model(sample_count, initial_weights, empty_map=empty)
    np.random.seed(seed)
    result = marginalization_tools.DistMarg.draw_sky_times(model, snrs)
    next_random = np.random.random()
    return model, result, next_random, leaves


def test_draw_sky_times_preserves_public_gps_and_rng(torch_device):
    torch = pytest.importorskip("torch")
    sample_count = 29
    expected_model, expected, expected_next, _ = _draw_full_sky(
        "numpy", sample_count, 941
    )
    model, actual, actual_next, leaves = _draw_full_sky(
        "torch", sample_count, 941, device=torch_device
    )

    assert actual_next == expected_next
    assert model.sample_idx.tolist() == expected_model.sample_idx.tolist()
    assert set(model.sample_idx) == {0, 1}
    for key in ("tc", "ra", "dec"):
        assert type(actual[key]) is np.ndarray
        assert actual[key].dtype == np.float64
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual["logw_partial"].device.type == torch_device.type
    actual_logw = actual["logw_partial"].cpu()
    torch.testing.assert_close(
        actual_logw,
        torch.as_tensor(expected["logw_partial"], dtype=actual_logw.dtype),
        rtol=1e-5 if torch_device.type == "mps" else 1e-12,
        atol=2e-5 if torch_device.type == "mps" else 1e-12,
    )

    # Large absolute GPS coordinates never become float32 device values.
    assert np.max(
        np.abs(actual["tc"] - actual["tc"].astype(np.float32))
    ) > 0.1
    actual["logw_partial"].sum().backward()
    assert all(leaf.grad is not None for leaf in leaves)
    assert all(leaf.grad.device.type == torch_device.type for leaf in leaves)


def test_draw_sky_times_empty_delay_map_keeps_public_fallback():
    torch = pytest.importorskip("torch")
    expected_model, expected, expected_next, _ = _draw_full_sky(
        "numpy", 17, 664, empty=True
    )
    model, actual, actual_next, _ = _draw_full_sky(
        "torch", 17, 664, device=torch.device("cpu"), empty=True
    )

    assert expected is None
    assert actual is None
    assert actual_next == expected_next
    np.testing.assert_array_equal(
        model.marginalize_vector_params["logw_partial"],
        expected_model.marginalize_vector_params["logw_partial"],
    )
    assert type(model.marginalize_vector_params["logw_partial"]) is np.ndarray
