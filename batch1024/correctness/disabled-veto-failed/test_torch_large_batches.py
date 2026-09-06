# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Bounded correctness checks with 1,024 independent rows on real devices."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pycbc  # noqa: E402
from pycbc import scheme  # noqa: E402
from pycbc.types import Array, FrequencySeries, zeros  # noqa: E402

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without Torch support", allow_module_level=True)

BATCH = 1024


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    state, lock, single = scheme.mgr.state, scheme.mgr._lock, scheme.Scheme._single
    threads = torch.get_num_threads()
    for flag in (
        "CPU_FFTW_BATCH",
        "CPU_NATIVE_BATCH_CORRELATE",
        "CUDA_NATIVE_BATCH_CORRELATE",
        "CUDA_NATIVE_BATCH_PEAK",
        "CUDA_PROMOTED_ROWS",
        "DIRECT_BATCH_IFFT",
        "ONDEVICE_PEAKS",
    ):
        monkeypatch.setenv("PYCBC_TORCH_" + flag, "0")
    monkeypatch.delenv("PYCBC_BATCH_MAXELEMENTS", raising=False)
    torch.set_num_threads(1)
    try:
        yield
    finally:
        scheme.mgr._lock = False
        scheme.mgr.state = state
        scheme.Scheme._single = single
        scheme.mgr._lock = lock
        torch.set_num_threads(threads)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return request.param


def _complex_values(size, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(BATCH, size)) + 1j * rng.normal(size=(BATCH, size))
    ).astype(np.complex64)


@pytest.mark.parametrize("inverse", [False, True])
def test_complex_fft_rows(device, inverse):
    from pycbc.fft import torchfft

    size = 31
    values = _complex_values(size, 9101)
    expected = (
        np.fft.ifft(values, axis=-1) * size if inverse else np.fft.fft(values, axis=-1)
    )
    with scheme.TorchScheme(device):
        source, target = Array(values.ravel()), zeros(values.size, np.complex64)
        engine = (torchfft.IFFT if inverse else torchfft.FFT)(
            source,
            target,
            nbatch=BATCH,
            size=size,
        )
        engine.execute()
        np.testing.assert_array_equal(source.numpy(), values.ravel())
        assert target._data.tensor.device.type == device
        np.testing.assert_allclose(
            target.numpy().reshape(BATCH, size),
            expected,
            rtol=2e-6,
            atol=2e-6,
        )


def test_real_fft_row_boundaries(device):
    from pycbc.fft import torchfft

    size = 15
    values = _complex_values(size, 9102).real.copy()
    frequencies = _complex_values(size // 2 + 1, 9103)
    with scheme.TorchScheme(device):
        forward = zeros(frequencies.size, np.complex64)
        inverse = zeros(values.size, np.float32)
        torchfft.FFT(Array(values.ravel()), forward, nbatch=BATCH, size=size).execute()
        torchfft.IFFT(
            Array(frequencies.ravel()), inverse, nbatch=BATCH, size=size
        ).execute()
        assert forward._data.tensor.device.type == device
        assert inverse._data.tensor.device.type == device
        np.testing.assert_allclose(
            forward.numpy().reshape(BATCH, -1),
            np.fft.rfft(values, axis=-1),
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            inverse.numpy().reshape(BATCH, size),
            np.fft.irfft(frequencies, n=size, axis=-1) * size,
            rtol=2e-6,
            atol=2e-6,
        )


def test_promoted_ifft_final_partial_chunk(device, monkeypatch):
    from pycbc.fft import torchfft

    size, capacity = 64, 3  # 1,024 leaves one row in the final chunk.
    monkeypatch.setitem(torchfft._PROMOTED_BATCH_MAX_ELEMENTS, device, capacity * size)
    values = _complex_values(size, 9104)
    expected = np.fft.ifft(values.astype(np.complex128), axis=-1) * size
    with scheme.TorchScheme(device):
        for inplace in (False, True):
            source = Array(values.ravel())
            target = source if inplace else zeros(values.size, np.complex64)
            engine = torchfft.IFFT(source, target, nbatch=BATCH, size=size)
            plan = engine._promoted_batch_plan
            assert plan.rows == capacity
            assert plan.source.shape == (capacity, size)
            assert plan.source.device.type == device
            engine.execute()
            np.testing.assert_allclose(
                target.numpy().reshape(BATCH, size),
                expected,
                rtol=5e-7,
                atol=5e-7,
            )


@pytest.mark.parametrize("inverse", [False, True])
def test_direct_cpu_fftw_reuses_full_batch(monkeypatch, inverse):
    from pycbc.fft import fftw, torchfft

    if not torchfft._FFTW_DIRECT_PLATFORM_SUPPORTED:
        pytest.skip("direct FFTW requires its supported Linux/x86-64 platform")
    monkeypatch.setenv("PYCBC_TORCH_CPU_FFTW_BATCH", "1")
    # Exercise the real plan_many ABI with bounded buffers, as in the focused
    # native FFT tests; platform and library eligibility remain authoritative.
    monkeypatch.setattr(torchfft, "_FFTW_DIRECT_BATCH_SIZES", frozenset({64}))
    old_measure = fftw.get_measure_level()
    fftw.set_measure_level(0)
    try:
        inputs = [_complex_values(64, seed) for seed in (9105, 9106)]
        references = []
        with scheme.CPUScheme(1):
            for values in inputs:
                source = zeros(values.size, np.complex64)
                output = zeros(values.size, np.complex64)
                source.data[:] = values.ravel()
                assert source.ptr % pycbc.PYCBC_ALIGNMENT == 0
                assert output.ptr % pycbc.PYCBC_ALIGNMENT == 0
                (fftw.IFFT if inverse else fftw.FFT)(
                    source,
                    output,
                    nbatch=BATCH,
                    size=64,
                ).execute()
                references.append(output.numpy().copy())
        with scheme.TorchScheme("cpu"):
            source, target = (
                zeros(inputs[0].size, np.complex64),
                zeros(inputs[0].size, np.complex64),
            )
            source._data.tensor.copy_(torch.from_numpy(inputs[0].ravel()))
            assert source.ptr % pycbc.PYCBC_ALIGNMENT == 0
            assert target.ptr % pycbc.PYCBC_ALIGNMENT == 0
            engine = (torchfft.IFFT if inverse else torchfft.FFT)(
                source,
                target,
                nbatch=BATCH,
                size=64,
            )
            plan = engine._fftw_batch_plan
            if plan is None:
                pytest.skip("single-precision direct FFTW plan is unavailable")
            assert plan._batch == BATCH
            assert engine._promoted_batch_plan is None
            pointers = (source.ptr, target.ptr)
            for values, expected in zip(inputs, references, strict=True):
                source._data.tensor.copy_(torch.from_numpy(values.ravel()))
                version = target._data.tensor._version
                engine.execute()
                assert engine._fftw_batch_plan is plan
                assert (source.ptr, target.ptr) == pointers
                assert target._data.tensor._version == version + 1
                np.testing.assert_array_equal(source.numpy(), values.ravel())
                np.testing.assert_array_equal(target.numpy(), expected)
    finally:
        fftw.set_measure_level(old_measure)


def test_native_correlation_row_tails_and_reuse(device, monkeypatch):
    from pycbc.filter import matchedfilter_cpu, matchedfilter_torch
    from pycbc.filter.matchedfilter import BatchCorrelator

    if device == "cpu":
        runtime = matchedfilter_torch._cpu_native_openmp_runtime(matchedfilter_cpu)
        if not matchedfilter_torch._cpu_native_batch_runtime_is_stable(runtime):
            pytest.skip("CPU correlation needs matching fixed Torch/OpenMP teams")
    monkeypatch.setenv(f"PYCBC_TORCH_{device.upper()}_NATIVE_BATCH_CORRELATE", "1")
    size, total = 129, 160  # Aligned rows, with sentinel tails beyond the kernel.
    values = _complex_values(total, 9107)
    data = _complex_values(total, 9108)[0]
    with scheme.TorchScheme(device):
        x, z, y = Array(values.ravel()), zeros(values.size, np.complex64), Array(data)
        z._data.tensor.fill_(19 - 7j)
        xs = [x[i * total : (i + 1) * total] for i in range(BATCH)]
        zs = [z[i * total : (i + 1) * total] for i in range(BATCH)]
        batch = BatchCorrelator(xs, zs, size)
        state_name = f"_torch_{device}_native_batch_state"
        state = None
        for iteration in range(2):
            if iteration:
                # Mutate both ends and a middle row without rebinding buffers.
                for index in (0, BATCH // 2, BATCH - 1):
                    xs[index]._data.tensor.mul_(2 - 0.5j)
                y._data.tensor.mul_(0.5 + 0.25j)
            if device == "cpu":
                expected = np.empty((BATCH, size), np.complex64)
                for index, row in enumerate(xs):
                    matchedfilter_cpu._correlate(
                        row.numpy()[:size],
                        y.numpy()[:size],
                        expected[index],
                    )
            else:
                expected = (
                    (
                        x._data.tensor.view(BATCH, total)[:, :size].conj()
                        * y._data.tensor[:size]
                    )
                    .cpu()
                    .numpy()
                )
            batch.execute(y)
            current = getattr(batch, state_name, None)
            assert current is not None, "eligible native route did not execute"
            assert current._num_vectors == BATCH
            if state is not None:
                assert current is state
            state = current
            result = z.numpy().reshape(BATCH, total)
            np.testing.assert_array_equal(result[:, :size], expected)
            np.testing.assert_array_equal(result[:, size:], np.complex64(19 - 7j))


def test_peak_ties_nan_overflow_and_slice_boundaries(device, monkeypatch):
    from pycbc.filter import matchedfilter, matchedfilter_torch

    cases = np.array(
        [
            [0, 3 + 4j, -3 - 4j, 1],
            [1, complex(np.nan, 0), 6, 5],
            [1, np.inf, -np.inf, 9],
            [complex(np.nan, 1), complex(np.nan, -2), 0, 0],
            [1e20 + 1e20j, -2e20, 3, 4],
            [complex(-0.0, 0), complex(0, -0.0), 0, 0],
        ],
        dtype=np.complex64,
    )
    case_ids = np.arange(BATCH) % len(cases)
    values = np.full((BATCH, 8), 300 + 400j, np.complex64)
    values[:, 2:6] = cases[case_ids]
    expected_indices = np.array([1, 2, 1, 0, 0, 0])[case_ids]
    expected_peaks = values[np.arange(BATCH), 2 + expected_indices]
    with scheme.TorchScheme(device):
        output = Array(values.ravel())
        if device == "cuda":
            monkeypatch.setenv("PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK", "1")
            calls = []
            original = matchedfilter_torch.standard_peak_tensor

            def observed(tensor):
                calls.append(tuple(tensor.shape))
                return original(tensor)

            monkeypatch.setattr(matchedfilter_torch, "standard_peak_tensor", observed)
            indices, peaks = matchedfilter._torch_batch_peak_values(
                output,
                BATCH,
                8,
                slice(2, 6),
            )
            assert calls == [(BATCH, 4)]
        else:
            # Main has no optional CPU peak extension. Exercise its real
            # legacy-semantics tensor reduction directly instead.
            indices, peaks = matchedfilter_torch.standard_peak_tensor(
                output._data.tensor.view(BATCH, 8)[:, 2:6],
            )
            indices, peaks = indices.numpy(), peaks.numpy()
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(
            peaks.view(np.uint64), expected_peaks.view(np.uint64)
        )


@pytest.mark.parametrize("native", [False, True], ids=["torch", "triton"])
def test_peak_threshold_equality_and_abort(device, native, monkeypatch):
    from pycbc.filter import matchedfilter_torch

    if native and (device != "cuda" or not matchedfilter_torch._HAS_TRITON):
        pytest.skip("native threshold kernel requires CUDA and Triton")
    monkeypatch.setenv("PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK", str(int(native)))
    calls = []
    if native:
        kernel = matchedfilter_torch._triton_batch_magsq_argmax_kernel
        original = kernel.run

        def observed(*args, **kwargs):
            calls.append(kwargs.get("grid"))
            return original(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", observed)
    # Every row is below, exactly at, or one float32 step above the threshold.
    amplitudes = np.array(
        [
            np.nextafter(np.float32(2), np.float32(0)),
            2,
            np.nextafter(np.float32(2), np.float32(3)),
        ]
    )
    row_types = np.arange(BATCH) % 3
    values = torch.zeros((BATCH, 8), dtype=torch.complex64, device=device)
    values[:, 2] = torch.as_tensor(amplitudes[row_types], device=device)
    values[:, 5] = -values[:, 2]  # Equal maxima must choose the first index.
    reduce = matchedfilter_torch._torch_batch_peak_and_threshold_gpu
    survivors, indices, peaks, aborted = reduce(values, np.ones(BATCH), 2.0)
    expected = np.flatnonzero(row_types != 0)
    np.testing.assert_array_equal(survivors, expected)
    np.testing.assert_array_equal(indices, 2)
    np.testing.assert_array_equal(peaks, amplitudes[row_types[expected]])
    assert not aborted
    assert not reduce(values, np.ones(BATCH), 2.0, float(amplitudes[-1]))[3]
    assert reduce(values, np.ones(BATCH), 2.0, 2.0)[3]
    if native:
        assert len(calls) == 3
        assert all(grid == (BATCH,) for grid in calls)


def test_public_live_filter_full_batch(device):
    from pycbc.filter import matchedfilter
    from pycbc.vetoes.sgchisq import SingleDetSGChisq

    size = 32
    values, data_values = _complex_values(17, 9109), _complex_values(17, 9110)[0]
    power = (values.real**2 + values.imag**2).sum(axis=1) * (4.0 / size)
    with scheme.TorchScheme(device):
        templates = []
        for index, row in enumerate(values):
            template = FrequencySeries(row, delta_f=1.0 / size)
            template.id = index
            template.params = np.array([(20.0,)], dtype=[("mass1", np.float32)])[0]
            template.sigmasq = lambda _psd, value=power[index]: value
            templates.append(template)
        batch = matchedfilter.LiveBatchMatchedFilter(
            templates,
            snr_threshold=0.0,
            chisq_bins=0,
            sg_chisq=SingleDetSGChisq(templates),
            maxelements=BATCH * size,
            enable_cuda_graphs=False,
            enable_async_streams=False,
        )
        data = FrequencySeries(data_values, delta_f=1.0 / size)
        data.psd = FrequencySeries(np.ones(17, np.float32), delta_f=1.0 / size)
        reader = SimpleNamespace(
            overwhitened_data=lambda _delta_f: data,
            trim_padding=0,
            blocksize=size,
            sample_rate=1,
            start_time=100.0,
        )
        mid = batch.mids[0]
        assert len(batch.tgroups) == 1
        assert batch.ifts[mid].nbatch == BATCH
        assert batch.ifts[mid].__class__.__module__ == "pycbc.fft.torchfft"
        row_ids = np.array([template.id for template in batch.tgroups[0]])
        pointers = [template.cout.ptr for template in templates]
        result = batch.process_data(reader)
        assert [template.cout.ptr for template in templates] == pointers
        np.testing.assert_array_equal(data.numpy(), data_values)
        np.testing.assert_array_equal(np.stack([t.numpy() for t in templates]), values)
        correlation = batch.cout_mem[mid].numpy().reshape(BATCH, size)
        output = batch.out_mem[mid].numpy().reshape(BATCH, size)
        expected_corr = np.zeros((BATCH, size), np.complex64)
        expected_corr[:, :17] = values.conj() * data_values
        expected = np.fft.ifft(expected_corr.astype(np.complex128), axis=-1) * size
        np.testing.assert_allclose(
            correlation[:, :17], expected_corr[row_ids, :17], rtol=2e-7, atol=2e-7
        )
        np.testing.assert_array_equal(correlation[:, 17:], 0)
        np.testing.assert_allclose(output, expected[row_ids], rtol=4e-7, atol=2e-6)
        assert (
            np.linalg.norm(output - expected[row_ids]) / np.linalg.norm(expected)
            < 1.5e-7
        )
        order = np.argsort(result["template_id"])
        indices = np.abs(expected).argmax(axis=1)
        peaks = expected[np.arange(BATCH), indices]
        np.testing.assert_array_equal(result["template_id"][order], np.arange(BATCH))
        np.testing.assert_array_equal(result["end_time"][order], 100 + indices)
        np.testing.assert_allclose(
            result["snr"][order],
            np.abs(peaks) * (4.0 / size) / np.sqrt(power),
            rtol=4e-7,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            result["coa_phase"][order], np.angle(peaks), rtol=4e-7, atol=2e-7
        )


def test_relative_summary_and_likelihood_batches(device):
    from pycbc.inference.models import relbin_torch

    with torch.device(device):
        first = torch.as_tensor(_complex_values(33, 9111), dtype=torch.complex128)
        first.requires_grad_(True)
        frequencies = torch.arange(33, dtype=torch.float64) * 0.25
        edges = torch.arange(0, 33, 4)
        bins = torch.stack((edges[:-1], edges[1:]), dim=1)
        second = torch.ones(33, dtype=torch.complex128)
        psd = torch.ones(33, dtype=torch.float64)
        a0, a1 = relbin_torch.summary_product(
            first, second, psd, frequencies, bins, 0.25
        )
        reference = first.detach().cpu().numpy()[:, :32].conj().reshape(BATCH, 8, 4)
        np.testing.assert_allclose(
            a0.detach().cpu(), reference.sum(axis=-1), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            a1.detach().cpu(),
            (reference * (np.arange(4) * 0.25)).sum(axis=-1),
            rtol=1e-12,
            atol=1e-12,
        )
        assert a0.shape == a1.shape == (BATCH, 8)
        assert a0.device.type == a1.device.type == device
        (a0.real.sum() + a1.imag.sum()).backward()
        assert torch.isfinite(first.grad).all()

        fp = torch.linspace(0.1, 1.0, BATCH, dtype=torch.float64)
        fc = torch.linspace(-0.5, 0.5, BATCH, dtype=torch.float64)
        delays = torch.linspace(-0.01, 0.01, BATCH, dtype=torch.float64)
        hp, hc = first.detach()[0, edges], first.detach()[1, edges]
        reference = torch.ones_like(hp)
        args = (
            hp,
            hc,
            reference,
            a0.detach()[0],
            a1.detach()[0],
            torch.ones(8, dtype=torch.float64),
            torch.zeros(8, dtype=torch.float64),
        )
        actual = relbin_torch.likelihood_parts(
            frequencies[edges], fp, fc, delays, *args
        )
        assert all(
            value.shape == (BATCH,) and value.device.type == device for value in actual
        )
        # Evaluate the same 1,024 samples independently through the scalar API.
        expected = tuple(
            torch.stack(values)
            for values in zip(
                *(
                    relbin_torch.likelihood_parts(
                        frequencies[edges], fp[i], fc[i], delays[i], *args
                    )
                    for i in range(BATCH)
                ),
                strict=True,
            )
        )
        for value, scalar in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, scalar)


@pytest.mark.parametrize("phase_marginalized", [False, True], ids=["gaussian", "phase"])
def test_public_gaussian_likelihood_and_gradient_batch(device, phase_marginalized):
    from pycbc.inference.models.gaussian_noise import GaussianNoise
    from pycbc.inference.models.marginalized_gaussian_noise import (
        MarginalizedPhaseGaussianNoise,
    )

    with scheme.TorchScheme(device):
        data = {
            det: FrequencySeries(
                np.full(129, 1e-23 + 1e-23j), delta_f=2.0, epoch=1126259460.0
            )
            for det in ("H1", "L1")
        }
        psds = {det: FrequencySeries(np.full(129, 1e-46), delta_f=2.0) for det in data}
        params = {
            name: torch.linspace(low, high, BATCH, dtype=torch.float64, device=device)
            for name, (low, high) in dict(
                ra=(1.1, 1.2),
                dec=(-0.3, -0.2),
                polarization=(0.2, 0.4),
                tc=(1126259462.0, 1126259462.001),
            ).items()
        }
        params["ra"].requires_grad_(True)
        model_class = (
            MarginalizedPhaseGaussianNoise if phase_marginalized else GaussianNoise
        )
        model = model_class(
            tuple(params),
            data,
            {"H1": 20.0, "L1": 24.0},
            psds=psds,
            static_params=dict(
                approximant="TaylorF2",
                mass1=30.0,
                mass2=20.0,
                distance=500.0,
                inclination=0.4,
                f_lower=20.0,
                coa_phase=0.2,
            ),
        )
        selected = [0, BATCH // 2, BATCH - 1]
        expected = []
        for index in selected:
            model.update(**{name: value[index] for name, value in params.items()})
            expected.append(model.loglr)
        previous_params, previous_stats = model._current_params, model._current_stats
        actual = model.batched_loglr(**params)
        assert actual.shape == (BATCH,) and actual.device.type == device
        assert torch.isfinite(actual).all()
        assert model._current_params is previous_params
        assert model._current_stats is previous_stats
        torch.testing.assert_close(
            actual[selected], torch.stack(expected), atol=1e-9, rtol=1e-10
        )
        torch.testing.assert_close(
            model.batched_loglikelihood(**params), actual + model.lognl
        )
        gradient = torch.autograd.grad(actual.sum(), params["ra"])[0]
        scalar_gradient = torch.autograd.grad(
            torch.stack(expected).sum(), params["ra"]
        )[0]
        assert gradient.shape == (BATCH,) and torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient[selected], scalar_gradient[selected])
