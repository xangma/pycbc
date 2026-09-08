"""Ordinary Torch searches preserve the unchanged CPU chi-square statistic."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import Array, FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("point_count", (1, 4))
@pytest.mark.parametrize("norm_type", (float, np.float32, np.float64))
@pytest.mark.parametrize("array_indices", (False, True))
def test_sparse_search_matches_unchanged_cpu(
    device, point_count, norm_type, array_indices, monkeypatch
):
    from pycbc.vetoes import chisq_cpu

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(73482)
    size = 2**18
    values = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        np.complex64
    )
    points = np.array([size - 3, 170131, 3217, 0], dtype=np.int64)[:point_count]
    snr = (rng.normal(size=point_count) + 1j * rng.normal(size=point_count)).astype(
        np.complex64
    )
    bins = (19, 1337, 16381, 16381, size - 21)
    norm = norm_type(0.117311)
    expected = power_chisq_at_points_from_precomputed(
        FrequencySeries(values, delta_f=0.125), snr, norm, bins, points
    )
    calls = []
    original = chisq_cpu.point_chisq_code

    def record_kernel(chisq, corr, n, length, shifts, edges, count):
        calls.append((chisq.dtype, shifts.dtype, corr.__array_interface__["data"][0]))
        return original(chisq, corr, n, length, shifts, edges, count)

    monkeypatch.setattr(chisq_cpu, "point_chisq_code", record_kernel)
    with scheme.TorchScheme(device):
        corr = FrequencySeries(values, delta_f=0.125)
        indices = Array(points) + 0 if array_indices else points
        snr_t = torch.as_tensor(snr, device=device)
        corr_before = corr._data.tensor.clone()
        snr_before = snr_t.clone()
        actual = power_chisq_at_points_from_precomputed(
            corr, TorchArrayData(snr_t), norm, bins, indices
        )._data.tensor
        assert actual.device.type == device
        assert actual.cpu().numpy().dtype == expected.dtype
        assert torch.equal(corr._data.tensor, corr_before)
        assert torch.equal(snr_t, snr_before)
        if device == "cpu":
            np.testing.assert_array_equal(actual.cpu().numpy(), expected)
            assert len(calls) == 1
            assert calls[0][:2] == (np.dtype(np.float32), np.dtype(np.float32))
            assert calls[0][2] == corr._data.tensor.data_ptr()
        else:
            np.testing.assert_allclose(
                actual.cpu().numpy(), expected, rtol=1e-4, atol=1e-5
            )


@pytest.mark.parametrize("kind", ("grad", "inference", "subclass", "negative"))
@pytest.mark.parametrize("field", ("corr", "snr", "points"))
def test_search_compatibility_rejects_special_storage(kind, field):
    from pycbc.vetoes.chisq_torch import _search_compat_point_chisq

    class SpecialTensor(torch.Tensor):
        pass

    values = dict(
        corr=torch.ones(32, dtype=torch.complex64),
        snr=torch.ones(1, dtype=torch.complex64),
        points=torch.ones(1, dtype=torch.float64),
    )
    if kind == "grad":
        values[field].requires_grad_(True)
    elif kind == "inference":
        with torch.inference_mode():
            values[field] = values[field].clone()
    elif kind == "subclass":
        values[field] = values[field].as_subclass(SpecialTensor)
    else:
        values[field] = torch._neg_view(values[field])
    assert _search_compat_point_chisq(
        values["corr"], values["points"], (0, 13, 31), values["snr"], 0.117311
    ) is None
