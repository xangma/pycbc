import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme
from pycbc.filter.qtransform import qplane, qseries, qtiling
from pycbc.filter import qtransform_torch
from pycbc.types import TimeSeries
from pycbc.types.array_torch import TorchArrayData


def _make_signal(dtype=np.float64):
    # Two seconds of data at 1024 Hz to give the q-transform enough bandwidth
    t = np.arange(0, 2, 1 / 1024.0)
    noise = np.random.default_rng(1234).normal(0.0, 0.05, t.size)
    sig = (
        noise + np.sin(2 * np.pi * 40 * t)
        + 0.5 * np.sin(2 * np.pi * 90 * t)
    ).astype(dtype)
    return sig, t[1] - t[0]


@pytest.fixture
def torch_ctx():
    ctx = scheme.TorchScheme("cpu")
    try:
        yield ctx
    finally:
        # Explicitly drop the singleton guard so other tests can create schemes
        del ctx
        scheme.Scheme._single = None


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_qtransform_torch_matches_cpu(torch_ctx, dtype):
    data, dt = _make_signal(dtype)

    # CPU reference
    with scheme.CPUScheme():
        ts_cpu = TimeSeries(data, delta_t=dt)
        t_cpu, f_cpu, plane_cpu = ts_cpu.qtransform(frange=(20, 120))

    # Torch path on CPU device
    with torch_ctx:
        ts_torch = TimeSeries(data, delta_t=dt)
        t_t, f_t, plane_t = ts_torch.qtransform(frange=(20, 120))

    assert isinstance(plane_t, torch.Tensor)
    assert plane_t.device.type == "cpu"
    assert plane_cpu.dtype == np.dtype(np.float64)
    assert plane_t.dtype == torch.float64
    diff = plane_t.cpu().numpy() - plane_cpu
    rel_l2 = np.linalg.norm(diff) / np.linalg.norm(plane_cpu)
    max_rel_l2 = 5e-6 if dtype == np.float32 else 1e-12
    assert rel_l2 < max_rel_l2
    assert np.allclose(t_t.cpu().numpy(), t_cpu)
    assert np.allclose(f_t.cpu().numpy(), f_cpu)


def test_qtransform_torch_interpolation_and_complex(torch_ctx):
    data, dt = _make_signal()
    with torch_ctx:
        ts = TimeSeries(data, delta_t=dt)
        t_t, f_t, plane_t = ts.qtransform(
            delta_t=dt / 2.0, delta_f=1.0, frange=(20, 120), return_complex=True
        )

    assert plane_t.is_complex()
    assert plane_t.shape[0] == len(f_t)
    assert plane_t.shape[1] == len(t_t)
    assert t_t.dtype == torch.float64
    assert f_t.dtype == t_t.dtype
    assert plane_t.dtype == torch.complex128
    assert torch.isfinite(plane_t).all()


def test_qtransform_batch_workspace_is_bounded(monkeypatch):
    row_limit = qtransform_torch._qseries_batch_row_limit

    # The production-scale 32-second/1024-Hz transform uses a 64-row batch,
    # while longer transforms automatically consume fewer rows.
    assert row_limit(32768, 135) == 64
    assert row_limit(65536, 135) == 32
    assert row_limit(32768, 16) == 16

    # Even a budget smaller than one row retains the scalar low-memory path.
    monkeypatch.setattr(
        qtransform_torch, "_QPLANE_BATCH_WORKSPACE_BYTES", 1
    )
    assert row_limit(32768, 135) == 1


@pytest.mark.parametrize(
    "values",
    (
        [3.0, 1.0, 2.0],
        [4.0, 1.0, 3.0, 2.0],
        [2.0, 2.0, 1.0, 3.0],
        [1.0, np.nan, 3.0],
        [-np.inf, 1.0, np.inf],
        [-np.inf, np.inf],
    ),
)
def test_qtransform_midpoint_median_matches_numpy(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    with np.errstate(invalid="ignore"):
        expected = np.median(np.asarray(values, dtype=np.float64))
    actual = qtransform_torch._midpoint_median(tensor)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=0)


def test_qtransform_midpoint_median_batches_without_sort(monkeypatch):
    values = torch.tensor(
        [[4.0, 1.0, 3.0, 2.0], [8.0, 6.0, 7.0, 5.0]],
        dtype=torch.float64,
    )

    def _reject_sort(*_args, **_kwargs):
        raise AssertionError("CPU midpoint median performed a full sort")

    monkeypatch.setattr(torch, "sort", _reject_sort)
    actual = qtransform_torch._midpoint_median(values)
    torch.testing.assert_close(
        actual, torch.tensor([2.5, 6.5], dtype=torch.float64), rtol=0, atol=0
    )


def test_qtransform_midpoint_median_preserves_tied_autograd_routing():
    values = torch.ones(4, dtype=torch.float64, requires_grad=True)
    expected_values = values.detach().clone().requires_grad_()

    actual = qtransform_torch._midpoint_median(values)
    ordered = torch.sort(expected_values).values
    expected = 0.5 * (ordered[1] + ordered[2])
    actual.backward()
    expected.backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        values.grad, expected_values.grad, rtol=0, atol=0
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS device is unavailable"
)
def test_qtransform_midpoint_median_mps_nonfinite_semantics():
    values = torch.tensor(
        [[4.0, 1.0, 3.0, 2.0], [1.0, np.nan, 3.0, 2.0]],
        device="mps",
        dtype=torch.float32,
    )
    actual = qtransform_torch._midpoint_median(values)
    torch.mps.synchronize()
    expected = np.median(values.cpu().numpy(), axis=-1)
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("return_complex", (False, True))
def test_qtransform_batch_matches_scalar_and_low_memory(
    torch_ctx, monkeypatch, dtype, return_complex
):
    data, dt = _make_signal(dtype)

    with torch_ctx:
        fseries = TimeSeries(data, delta_t=dt).to_frequencyseries()
        tiles = qtiling(fseries, (8, 8), (20, 120), mismatch=0.2)
        q = next(iter(tiles))
        frequencies = tiles[q][:7]
        expected = torch.stack([
            qseries(
                fseries, q, f0, return_complex=return_complex
            )._data.tensor
            for f0 in frequencies
        ])
        actual = qtransform_torch._qseries_batch(
            fseries, q, frequencies, return_complex=return_complex
        )

        monkeypatch.setattr(
            qtransform_torch, "_QPLANE_BATCH_WORKSPACE_BYTES", 1
        )
        low_memory = qtransform_torch._qseries_batch(
            fseries, q, frequencies, return_complex=return_complex
        )

    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(
        low_memory, expected, rtol=1e-13, atol=1e-13
    )


def test_qtransform_interpolation_orders_agree():
    values = torch.arange(35, dtype=torch.float64).reshape(5, 7)
    freqs_old = torch.linspace(20.0, 120.0, 5, dtype=torch.float64)
    times_old = torch.linspace(0.0, 2.0, 7, dtype=torch.float64)
    freqs_new = torch.linspace(20.0, 120.0, 11, dtype=torch.float64)
    times_new = torch.linspace(0.0, 2.0, 3, dtype=torch.float64)

    # These shapes choose time-first.  Transposing the problem chooses the
    # frequency-first branch and must yield the separable transpose result.
    time_first = qtransform_torch._bilinear_interp(
        values, freqs_old, times_old, freqs_new, times_new
    )
    frequency_first = qtransform_torch._bilinear_interp(
        values.T, times_old, freqs_old, times_new, freqs_new
    ).T
    torch.testing.assert_close(
        time_first, frequency_first, rtol=1e-15, atol=1e-15
    )


def test_qtransform_torch_cpu_promotes_single_precision_fft(
    torch_ctx, monkeypatch
):
    data, dt = _make_signal(np.float32)
    fft_inputs = []
    original_to_frequencyseries = TimeSeries.to_frequencyseries

    def _record_fft_dtype(series, *args, **kwargs):
        fft_inputs.append((series.dtype, series._data.tensor.device.type))
        return original_to_frequencyseries(series, *args, **kwargs)

    with torch_ctx:
        monkeypatch.setattr(
            TimeSeries, "to_frequencyseries", _record_fft_dtype
        )
        source = TimeSeries(data, delta_t=dt)
        actual = source.qtransform(frange=(20, 120))
        reference = TimeSeries(
            data.astype(np.float64), delta_t=dt
        ).qtransform(frange=(20, 120))

    assert source.dtype == np.dtype(np.float32)
    assert fft_inputs == [
        (np.dtype(np.float64), "cpu"),
        (np.dtype(np.float64), "cpu"),
    ]
    for actual_value, reference_value in zip(actual, reference):
        torch.testing.assert_close(
            actual_value, reference_value, rtol=0, atol=0
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device is unavailable"
)
def test_qtransform_cuda_promotes_single_precision_fft(monkeypatch):
    data, dt = _make_signal(np.float32)

    with scheme.CPUScheme():
        source = TimeSeries(data, delta_t=dt)
        expected_times, expected_freqs, expected_plane = source.qtransform(
            delta_t=4 * dt,
            delta_f=1.0,
            frange=(20, 120),
            qrange=(4, 64),
            mismatch=0.15,
        )

    fft_inputs = []
    original_to_frequencyseries = TimeSeries.to_frequencyseries

    def _record_fft_dtype(series, *args, **kwargs):
        fft_inputs.append(
            (series.dtype, series._data.tensor.device.type)
        )
        return original_to_frequencyseries(series, *args, **kwargs)

    def _reject_host_transfer(_self):
        raise AssertionError("CUDA Q-transform copied Torch data to host")

    ctx = scheme.TorchScheme("cuda")
    try:
        with ctx:
            monkeypatch.setattr(
                TimeSeries, "to_frequencyseries", _record_fft_dtype
            )
            monkeypatch.setattr(
                TorchArrayData, "numpy", _reject_host_transfer
            )
            source = TimeSeries(data, delta_t=dt)
            actual_times, actual_freqs, actual_plane = source.qtransform(
                delta_t=4 * dt,
                delta_f=1.0,
                frange=(20, 120),
                qrange=(4, 64),
                mismatch=0.15,
            )
            complex_source = TimeSeries(
                data.astype(np.complex64), delta_t=dt
            )
            with pytest.raises(
                TypeError, match="to_frequencyseries does not support complex"
            ):
                complex_source.qtransform(frange=(20, 120))
        torch.cuda.synchronize()
    finally:
        del ctx
        scheme.Scheme._single = None

    assert source.dtype == np.dtype(np.float32)
    assert source._data.tensor.device.type == "cuda"
    torch.testing.assert_close(
        source._data.tensor,
        torch.as_tensor(data, device="cuda"),
        rtol=0,
        atol=0,
    )
    assert fft_inputs == [
        (np.dtype(np.float64), "cuda"),
        (np.dtype(np.complex64), "cuda"),
    ]
    assert actual_times.device.type == "cuda"
    assert actual_freqs.device.type == "cuda"
    assert actual_plane.device.type == "cuda"
    assert actual_plane.dtype == torch.float64
    np.testing.assert_allclose(actual_times.cpu().numpy(), expected_times)
    np.testing.assert_allclose(actual_freqs.cpu().numpy(), expected_freqs)
    actual_plane = actual_plane.cpu().numpy()
    np.testing.assert_allclose(
        actual_plane, expected_plane, rtol=7e-5, atol=1e-8
    )
    relative_l2 = (
        np.linalg.norm(actual_plane - expected_plane)
        / np.linalg.norm(expected_plane)
    )
    assert relative_l2 < 1e-6


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS device is unavailable"
)
def test_qtransform_mps_uses_supported_single_precision():
    data, dt = _make_signal(np.float32)
    ctx = scheme.TorchScheme("mps")
    try:
        with ctx:
            source = TimeSeries(data, delta_t=dt)
            fseries = source.to_frequencyseries()
            tiles = qtiling(fseries, (8, 8), (20, 120), mismatch=0.2)
            series = qseries(fseries, 8, 40)
            _, times, freqs, plane = qplane(tiles, fseries)
            public_times, public_freqs, public_plane = source.qtransform(
                frange=(20, 120), qrange=(8, 8)
            )
            linear_times, linear_freqs, linear_plane = source.qtransform(
                delta_t=2 * dt,
                delta_f=2.0,
                frange=(20, 120),
                qrange=(8, 8),
            )
            log_times, log_freqs, log_plane = source.qtransform(
                delta_t=2 * dt,
                logfsteps=16,
                frange=(20, 120),
                qrange=(8, 8),
            )
        torch.mps.synchronize()
    finally:
        del ctx
        scheme.Scheme._single = None

    assert series._data.tensor.dtype == torch.float32
    assert times.dtype == torch.float32
    assert freqs.dtype == times.dtype
    assert plane.dtype == torch.float32
    assert torch.isfinite(plane).all()
    assert public_times.dtype == torch.float32
    assert public_freqs.dtype == public_times.dtype
    assert public_plane.dtype == torch.float32
    assert torch.isfinite(public_plane).all()
    assert linear_times.dtype == torch.float32
    assert linear_freqs.dtype == linear_times.dtype
    assert linear_plane.dtype == torch.float32
    assert linear_plane.shape == (len(linear_freqs), len(linear_times))
    assert torch.isfinite(linear_plane).all()
    assert log_times.dtype == torch.float32
    assert log_freqs.dtype == log_times.dtype
    assert log_plane.dtype == torch.float32
    assert log_plane.shape == (len(log_freqs), len(log_times))
    assert torch.isfinite(log_plane).all()
    assert torch.all(log_freqs[1:] > log_freqs[:-1])
    np.testing.assert_allclose(
        log_freqs[[0, -1]].cpu().numpy(), (20.0, 120.0), rtol=1e-6
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("return_complex", (False, True))
def test_direct_qtransform_functions_stay_on_torch(
    torch_ctx, monkeypatch, return_complex, dtype
):
    data, dt = _make_signal(dtype)

    with scheme.CPUScheme():
        cpu_fseries = TimeSeries(data, delta_t=dt).to_frequencyseries()
        tiles = qtiling(cpu_fseries, (8, 8), (20, 120), mismatch=0.2)
        expected_series = qseries(cpu_fseries, 8, 40, return_complex=return_complex)
        expected_q, expected_times, expected_freqs, expected_plane = qplane(
            tiles, cpu_fseries, return_complex=return_complex
        )

    with torch_ctx:
        torch_fseries = TimeSeries(data, delta_t=dt).to_frequencyseries()
        original_item = torch.Tensor.item
        item_calls = 0

        def _reject_host_transfer(_self):
            raise AssertionError("Q-transform copied Torch data to host")

        def _count_scalar_transfer(tensor, *args, **kwargs):
            nonlocal item_calls
            item_calls += 1
            if item_calls > 1:
                raise AssertionError(
                    "Q-transform synchronized more than once for plane selection"
                )
            return original_item(tensor, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(torch.Tensor, "item", _count_scalar_transfer)
            actual_series = qseries(torch_fseries, 8, 40, return_complex=return_complex)
            actual_q, actual_times, actual_freqs, actual_plane = qplane(
                tiles, torch_fseries, return_complex=return_complex
            )

    assert item_calls == 1
    assert actual_series._data.tensor.device.type == "cpu"
    assert expected_series.dtype == np.dtype(
        np.complex128 if return_complex else np.float64
    )
    assert actual_series._data.tensor.dtype == (
        torch.complex128 if return_complex else torch.float64
    )
    assert isinstance(actual_times, torch.Tensor)
    assert isinstance(actual_freqs, torch.Tensor)
    assert isinstance(actual_plane, torch.Tensor)
    assert actual_plane.device.type == "cpu"
    assert actual_times.dtype == torch.float64
    assert actual_freqs.dtype == actual_times.dtype
    assert actual_plane.dtype == (
        torch.complex128 if return_complex else torch.float64
    )
    assert actual_q == expected_q
    np.testing.assert_allclose(actual_times.cpu().numpy(), expected_times)
    np.testing.assert_allclose(actual_freqs.cpu().numpy(), expected_freqs)
    rtol = 1e-6 if dtype == np.float32 else 1e-10
    atol = 1e-8 if dtype == np.float32 else 1e-11
    np.testing.assert_allclose(
        actual_series._data.tensor.cpu().numpy(),
        expected_series.numpy(),
        rtol=rtol,
        atol=atol,
    )
    actual_plane_np = actual_plane.cpu().numpy()
    if return_complex:
        np.testing.assert_allclose(
            actual_plane_np, expected_plane, rtol=rtol, atol=atol
        )
    else:
        relative_l2 = (
            np.linalg.norm(actual_plane_np - expected_plane)
            / np.linalg.norm(expected_plane)
        )
        max_rel_l2 = 5e-6 if dtype == np.float32 else 1e-12
        assert relative_l2 < max_rel_l2
