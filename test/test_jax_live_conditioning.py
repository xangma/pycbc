"""Live conditioning retains JAX buffers through PSD and FFT operations."""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
from pycbc import scheme
from pycbc.strain import strain as strain_module
from pycbc.strain.strain import StrainBuffer, execute_cached_fft, execute_cached_ifft
from pycbc.types import TimeSeries
from pycbc.types.array_jax import to_jax


def _devices():
    devices = ['cpu']
    try:
        if jax.devices('gpu'):
            devices.append('cuda:0')
    except RuntimeError:
        pass
    return devices


def _buffer():
    sample_rate = 2048
    length = 8 * sample_rate
    rng = np.random.default_rng(20260920)
    obj = object.__new__(StrainBuffer)
    obj.strain = TimeSeries(rng.normal(size=length).astype(np.float32),
                           delta_t=1.0 / sample_rate, epoch=1000000000)
    obj.psd = None
    obj.psds = {}
    obj.segments = {}
    obj.sample_rate = sample_rate
    obj.reduced_pad = 0
    obj.trim_padding = int(0.25 * sample_rate)
    obj.psd_inverse_length = 1.5
    obj.low_frequency_cutoff = 30
    obj.psd_samples = 3
    obj.psd_segment_length = 1
    obj.psd_recalculate_difference = None
    obj.psd_abort_difference = None
    obj.detector = 'H1'
    obj.psd_duration = 1
    obj.recalculate_psd()
    return obj


def _forbid_native(*args, **kwargs):
    raise AssertionError('JAX conditioning entered the native FFT cache')


@pytest.mark.parametrize('device', _devices())
def test_live_psd_preserves_input_precision(device):
    from pycbc.psd import welch

    with scheme.JAXScheme(device):
        buffer = _buffer()
        segment_length = int(buffer.sample_rate * buffer.psd_segment_length)
        count = (buffer.psd_samples + 1) * segment_length // 2
        expected = welch(buffer.strain[len(buffer.strain)-count:],
                         seg_len=segment_length,
                         seg_stride=segment_length // 2)
        assert buffer.psd.dtype == np.float32
        np.testing.assert_array_equal(np.asarray(buffer.psd),
                                      np.asarray(expected))


@pytest.mark.parametrize('device', _devices())
def test_frame_buffer_accepts_floating_sample_rate(device, monkeypatch):
    from pycbc.frame.frame import DataBuffer

    def update_cache(self):
        self.stream = None

    monkeypatch.setattr(DataBuffer, 'update_cache', update_cache)
    monkeypatch.setattr(DataBuffer, '_retrieve_metadata',
                        staticmethod(lambda stream, channel: (None, 4096.0)))
    with scheme.JAXScheme(device) as ctx:
        buffer = DataBuffer([], 'H1:STRAIN', 1000000000,
                            max_buffer=0.5, dtype=np.float32)
        assert len(buffer.raw_buffer) == 2048
        assert buffer.raw_buffer.delta_t == 1 / 4096
        assert float(buffer.raw_buffer.start_time) == 999999999.5
        assert to_jax(buffer.raw_buffer).devices() == {ctx.jax_device}
        np.testing.assert_array_equal(buffer.raw_buffer.numpy(), 0)


@pytest.mark.parametrize('device', _devices())
@pytest.mark.parametrize('allocator', ['zeros', 'empty'])
def test_jax_allocator_preserves_native_scalar_length(device, allocator):
    from pycbc import types
    allocate = getattr(types, allocator)
    with scheme.CPUScheme():
        expected = allocate(17.75, dtype=np.complex64)
    with scheme.JAXScheme(device) as ctx:
        result = allocate(17.75, dtype=np.complex64)
        assert result.shape == expected.shape == (17,)
        assert result.dtype == expected.dtype
        assert to_jax(result).devices() == {ctx.jax_device}


@pytest.mark.parametrize('device', _devices())
@pytest.mark.parametrize('reduced_pad,trim_padding', [
    (0, 512), (64, 1), (64, 512), (64, 513), (64, 681),
])
def test_live_overwhitening_on_selected_device(
        device, reduced_pad, trim_padding, monkeypatch):
    with scheme.CPUScheme():
        reference = _buffer()
        reference.reduced_pad = reduced_pad
        reference.trim_padding = trim_padding
        ref = reference.overwhitened_data(0.5)
        ref_values, ref_psd = ref.numpy().copy(), ref.psd.numpy().copy()

    monkeypatch.setattr(strain_module,
                        'create_memory_and_engine_for_class_based_fft',
                        _forbid_native)
    with scheme.JAXScheme(device) as ctx:
        buffer = _buffer()
        buffer.reduced_pad = reduced_pad
        buffer.trim_padding = trim_padding
        result = buffer.overwhitened_data(0.5)
        assert result.dtype == ref.dtype
        assert result.delta_f == ref.delta_f
        assert result.start_time == ref.start_time
        for series in (buffer.strain, buffer.psd, result, result.psd,
                       result.psd.psdt):
            assert isinstance(series.data.array, jax.Array)
            assert to_jax(series).devices() == {ctx.jax_device}
        values, psd = np.asarray(result), np.asarray(result.psd)
    # Whole-vector error avoids dividing by individual near-zero FFT bins.
    assert np.linalg.norm(values - ref_values) / np.linalg.norm(ref_values) < 3e-6
    # Inverse truncation suppresses the stop band almost to zero before
    # taking its reciprocal. Compare PSDs in band and the whitening transfer
    # function everywhere, avoiding ill-conditioned stop-band reciprocals.
    kmin = int(buffer.low_frequency_cutoff / result.delta_f)
    # A float32 PSD passes through several FFTs, a square and a reciprocal;
    # allow 5 ppm between FFT libraries while retaining the tighter overall
    # whitening error above. This does not change the benchmark science gates.
    np.testing.assert_allclose(psd[kmin:], ref_psd[kmin:], rtol=5e-6)
    np.testing.assert_allclose(1.0 / psd, 1.0 / ref_psd,
                               rtol=3e-6, atol=3e-6 / ref_psd.min())


@pytest.mark.parametrize('device', _devices())
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('trim_padding', [8191, 8192, 8193, 8194, 16385])
def test_live_overwhitening_clips_oversized_taper(
        device, dtype, trim_padding, monkeypatch):
    from pycbc.types import FrequencySeries

    sample_rate, output_length, reduced_pad = 2048, 4096, 64
    delta_f = sample_rate / output_length
    data = np.random.default_rng(4).normal(size=8192).astype(dtype)

    def buffer():
        obj = object.__new__(StrainBuffer)
        obj.strain = TimeSeries(data.copy(), delta_t=1 / sample_rate,
                               epoch=1000000000)
        obj.sample_rate = sample_rate
        obj.reduced_pad = reduced_pad
        obj.trim_padding = trim_padding
        obj.segments = {}
        psd = FrequencySeries(np.ones(output_length // 2 + 1, dtype=dtype),
                              delta_f=delta_f)
        padded_length = output_length + 2 * reduced_pad
        psd.psdt = FrequencySeries(
            np.ones(padded_length // 2 + 1, dtype=dtype),
            delta_f=sample_rate / padded_length)
        obj.psds = {delta_f: psd}
        return obj

    with scheme.CPUScheme():
        reference = buffer().overwhitened_data(delta_f)
        expected = reference.numpy().copy()

    monkeypatch.setattr(strain_module,
                        'create_memory_and_engine_for_class_based_fft',
                        _forbid_native)
    with scheme.JAXScheme(device) as ctx:
        obj = buffer()
        result = obj.overwhitened_data(delta_f)
        assert result is obj.overwhitened_data(delta_f)
        assert result.dtype == reference.dtype
        assert result.delta_f == reference.delta_f
        assert result.start_time == reference.start_time
        assert to_jax(result).devices() == {ctx.jax_device}
        error = np.linalg.norm(result.numpy() - expected)
        tolerance = 3e-6 if dtype == np.float32 else 1e-12
        assert error / np.linalg.norm(expected) < tolerance


@pytest.mark.parametrize('device', _devices())
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cached_fft_roundtrip_across_scheme_contexts(device, dtype, monkeypatch):
    samples = np.random.default_rng(81).normal(size=2048).astype(dtype)
    with scheme.CPUScheme():
        ref = execute_cached_fft(TimeSeries(samples, delta_t=1/2048, epoch=123))
        expected = ref.numpy().copy()
    monkeypatch.setattr(strain_module,
                        'create_memory_and_engine_for_class_based_fft',
                        _forbid_native)
    with scheme.JAXScheme(device) as ctx:
        source = TimeSeries(samples, delta_t=1/2048, epoch=123)
        result = execute_cached_fft(source)
        restored = execute_cached_ifft(result)
        assert to_jax(result).devices() == {ctx.jax_device}
        assert to_jax(restored).devices() == {ctx.jax_device}
        assert restored.dtype == dtype
        assert restored.delta_t == source.delta_t
        assert restored.start_time == source.start_time
        tol = 2e-6 if dtype == np.float32 else 2e-14
        np.testing.assert_allclose(result.numpy(), expected, rtol=tol, atol=tol)
        np.testing.assert_allclose(restored.numpy(), samples, rtol=tol, atol=tol)


@pytest.mark.parametrize('device', _devices())
def test_zero_corruption_does_not_erase_autogating_signal(device):
    from pycbc.strain.strain_jax import _whiten_pad_core
    samples = np.zeros(128, dtype=np.float32)
    samples[64] = 10
    with scheme.JAXScheme(device):
        import jax.numpy as jnp
        magnitude = _whiten_pad_core(
            jnp.asarray(samples), jnp.ones(65, dtype=jnp.float32),
            1.0, 128, 0, 128, 0, 0, 65)
        np.testing.assert_allclose(np.asarray(magnitude), samples, atol=1e-6)


@pytest.mark.parametrize('device', _devices())
@pytest.mark.parametrize('factor', [1, 4])
def test_live_block_conditioning_fusion_matches_existing_jax_path(
        device, factor, monkeypatch):
    from pycbc.frame.frame import DataBuffer
    from pycbc.filter import highpass_fir, resample_to_delta_t

    sample_rate = 2048
    blocksize = 2
    sample_step = blocksize * sample_rate
    corruption = 42
    highpass_samples = 128
    dynamic_range_factor = 2.5
    rng = np.random.default_rng(20260928)

    with scheme.JAXScheme(device) as ctx:
        raw = TimeSeries(rng.normal(size=8192 * factor),
                         delta_t=1 / (sample_rate * factor),
                         epoch=1000000000)
        initial = TimeSeries(
            rng.normal(size=16384).astype(np.float32),
            delta_t=1 / sample_rate,
            epoch=999999992,
        )
        expected = initial.copy()
        conditioned_size = sample_step + 2 * corruption
        block = raw[len(raw) - conditioned_size * factor:]
        block = highpass_fir(block, 25, highpass_samples, beta=5)
        block = (block * dynamic_range_factor).astype(np.float32)
        block = resample_to_delta_t(
            block, 1 / sample_rate, method='ldas')[corruption:]
        expected.roll(-sample_step)
        expected[len(expected) - conditioned_size + corruption:] = block[:]
        expected.start_time += blocksize

        buffer = object.__new__(StrainBuffer)
        buffer.raw_buffer = raw
        buffer.strain = initial
        buffer.factor = factor
        buffer.corruption = corruption
        buffer.sample_rate = sample_rate
        buffer.highpass_frequency = 25
        buffer.highpass_samples = highpass_samples
        buffer.beta = 5
        buffer.dyn_range_fac = dynamic_range_factor
        buffer.taper_immediate_strain = False
        buffer.state = buffer.dq = buffer.idq = None
        buffer.wait_duration = blocksize
        buffer.autogating_threshold = None
        buffer.psd = object()
        buffer.detector = 'H1'
        monkeypatch.setattr(
            DataBuffer,
            'attempt_advance',
            lambda self, size, timeout=10:
                raw[len(raw) - int(size / raw.delta_t):],
        )
        assert buffer.advance(blocksize)
        assert buffer._jax_conditioning_cache is not None

        assert to_jax(buffer.strain).devices() == {ctx.jax_device}
        assert buffer.strain.start_time == expected.start_time
        np.testing.assert_array_equal(np.asarray(buffer.strain),
                                      np.asarray(expected))


@pytest.mark.parametrize('device', _devices())
@pytest.mark.parametrize('invalid_psd', [0.0, -1.0, np.nan])
@pytest.mark.parametrize('in_band', [False, True])
def test_whitening_preserves_nonpositive_psd_semantics(
        device, invalid_psd, in_band):
    from pycbc.strain.strain_jax import _whiten_pad_core
    import jax.numpy as jnp
    samples = np.zeros(128, dtype=np.float32)
    samples[64] = 1
    psd = np.ones(65, dtype=np.float32)
    psd[3 if in_band else 0] = invalid_psd
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_asd = (psd * np.float32(1.0)) ** (-0.5)
        inv_asd[:2] = 0
        inv_asd[64:] = 0
        expected = np.abs(np.fft.irfft(
            np.fft.rfft(samples) * inv_asd,
            n=128))
    with scheme.JAXScheme(device) as ctx:
        got = _whiten_pad_core(
            jnp.asarray(samples), jnp.asarray(psd), 1.0,
            128, 0, 128, 0, 2, 64)
        assert got.devices() == {ctx.jax_device}
        assert got.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(got), expected, atol=1e-7,
                                   equal_nan=True)


@pytest.mark.parametrize('device', _devices())
@pytest.mark.parametrize('trim_padding', [512, 513])
def test_multi_duration_overwhitened_data_caching_and_parity(
        device, trim_padding):
    """Multi-duration overwhitening batches device kernels and caches."""
    sample_rate = 2048
    length = 32 * sample_rate
    rng = np.random.default_rng(20260930)
    data = rng.normal(size=length).astype(np.float32)

    durations = [8.0, 4.0, 2.0]
    dfs = tuple(1.0 / d for d in durations)

    with scheme.CPUScheme():
        ref_buf = object.__new__(StrainBuffer)
        ref_buf.strain = TimeSeries(
            data.copy(), delta_t=1.0 / sample_rate, epoch=1000000000,
        )
        ref_buf.sample_rate = sample_rate
        ref_buf.reduced_pad = 64
        ref_buf.trim_padding = trim_padding
        ref_buf.psd_inverse_length = 1.5
        ref_buf.low_frequency_cutoff = 30
        ref_buf.psd_samples = 3
        ref_buf.psd_segment_length = 1
        ref_buf.psd_recalculate_difference = None
        ref_buf.psd_abort_difference = None
        ref_buf.detector = 'H1'
        ref_buf.psd_duration = 1
        ref_buf.psd = None
        ref_buf.psds = {}
        ref_buf.segments = {}
        ref_buf.recalculate_psd()

        cpu_refs = {df: ref_buf.overwhitened_data(df).numpy().copy()
                    for df in dfs}

    with scheme.JAXScheme(device) as ctx:
        buffer = object.__new__(StrainBuffer)
        buffer.strain = TimeSeries(
            data.copy(), delta_t=1.0 / sample_rate, epoch=1000000000,
        )
        buffer.sample_rate = sample_rate
        buffer.reduced_pad = 64
        buffer.trim_padding = trim_padding
        buffer.psd_inverse_length = 1.5
        buffer.low_frequency_cutoff = 30
        buffer.psd_samples = 3
        buffer.psd_segment_length = 1
        buffer.psd_recalculate_difference = None
        buffer.psd_abort_difference = None
        buffer.detector = 'H1'
        buffer.psd_duration = 1
        buffer.psd = None
        buffer.psds = {}
        buffer.segments = {}
        buffer.recalculate_psd()
        buffer.required_delta_fs = dfs

        # Calling for the first delta_f fuses and preloads all required_delta_fs
        first_result = buffer.overwhitened_data(dfs[0])
        assert to_jax(first_result).devices() == {ctx.jax_device}
        # All required_delta_fs are now cached in segments
        for df in dfs:
            assert df in buffer.segments
            res = buffer.overwhitened_data(df)
            assert res is buffer.segments[df]
            val = np.asarray(res)
            ref_val = cpu_refs[df]
            rel_err = np.linalg.norm(val - ref_val) / np.linalg.norm(ref_val)
            assert rel_err < 3e-6, f"df {df} failed parity: rel_err={rel_err}"

        # Invalidate segments and test preload_overwhitened_data
        buffer.segments = {}
        preloaded = buffer.preload_overwhitened_data(dfs)
        assert len(preloaded) == len(dfs)
        for df in dfs:
            assert df in buffer.segments
            val = np.asarray(buffer.segments[df])
            ref_val = cpu_refs[df]
            rel_err = np.linalg.norm(val - ref_val) / np.linalg.norm(ref_val)
            assert rel_err < 3e-6


@pytest.fixture
def cuda_psd_device():
    if 'cuda:0' not in _devices():
        pytest.skip('CUDA JAX device unavailable')
    return 'cuda:0'


def _psd_preparation_buffer(dtype=np.float32, reduced_pad=64, cutoff=30,
                            psd_values=None):
    """Nonlinear source PSD on a coarser grid than the requested durations."""
    from pycbc.types import FrequencySeries

    sample_rate = 2048
    obj = object.__new__(StrainBuffer)
    data = np.random.default_rng(8052).normal(size=16 * sample_rate)
    obj.strain = TimeSeries(data.astype(dtype), delta_t=1 / sample_rate,
                           epoch=1000000000)
    obj.sample_rate = sample_rate
    obj.reduced_pad = reduced_pad
    obj.trim_padding = 513
    obj.psd_inverse_length = 1.5
    obj.low_frequency_cutoff = cutoff
    if psd_values is None:
        grid = np.arange(sample_rate // 2 + 1, dtype=np.float64)
        psd_values = (1.0 + 0.13 * np.sin(0.31 * grid)
                      + 0.003 * grid).astype(dtype)
    obj.psd = FrequencySeries(psd_values, delta_f=1.0, epoch=987654321)
    obj.psds = {}
    obj.segments = {}
    return obj


def _assert_psd_pair_bytes(actual, expected, device):
    for result, reference in ((actual, expected),
                              (actual.psdt, expected.psdt)):
        assert result.dtype == reference.dtype
        assert result.shape == reference.shape
        assert result.delta_f == reference.delta_f
        assert result.start_time == reference.start_time
        np.testing.assert_array_equal(
            np.asarray(result).view(np.uint8),
            np.asarray(reference).view(np.uint8),
        )
        assert to_jax(result).devices() == {device}
    assert actual._jax_psd is to_jax(actual)
    assert actual._jax_psdt is to_jax(actual.psdt)


def _record_psd_preparation(monkeypatch):
    from pycbc.strain import strain_jax as backend

    calls = []
    original = backend._multi_psd_prepare_core

    def record(psd, old_df, grid_dfs, static_configs):
        calls.append((tuple(grid_dfs), tuple(static_configs)))
        return original(psd, old_df, grid_dfs, static_configs)

    monkeypatch.setattr(backend, '_multi_psd_prepare_core', record)
    return calls


@pytest.mark.parametrize('dtype,reduced_pad,cutoff', [
    (np.float32, 0, None), (np.float64, 64, None),
    (np.float32, 64, 0), (np.float64, 0, 0),
    (np.float32, 64, 30), (np.float64, 0, 30),
    (np.float32, 0, 0.1), (np.float64, 64, 0.1),
])
def test_cuda_psd_preparation_matches_existing_jax_bytes(
        cuda_psd_device, dtype, reduced_pad, cutoff, monkeypatch):
    """Retain interpolation casts, hard truncation, and cutoff truthiness."""
    from pycbc.strain import strain_jax as backend

    dfs = (0.5, 0.25, 0.125)
    with scheme.JAXScheme(cuda_psd_device) as ctx:
        reference = _psd_preparation_buffer(dtype, reduced_pad, cutoff)
        for df in dfs:
            backend._ensure_psd_for_delta_f(reference, df)
        buffer = _psd_preparation_buffer(dtype, reduced_pad, cutoff)
        calls = _record_psd_preparation(monkeypatch)
        assert backend._prepare_missing_psds_cuda(buffer, dfs)
        assert tuple(buffer.psds) == dfs
        assert len(calls) == 1
        assert len(calls[0][0]) == len(calls[0][1]) == 2 * len(dfs)
        assert calls[0][0][1::2] == dfs
        for df in dfs:
            _assert_psd_pair_bytes(buffer.psds[df], reference.psds[df],
                                   ctx.jax_device)


def test_cuda_psd_preparation_hydrates_hits_and_refreshes_source(
        cuda_psd_device, monkeypatch):
    """Only missing grids prepare; invalidated caches use the current PSD."""
    from pycbc.strain import strain_jax as backend

    dfs = (0.5, 0.25, 0.125)
    with scheme.JAXScheme(cuda_psd_device) as ctx:
        buffer = _psd_preparation_buffer()
        cached = backend._ensure_psd_for_delta_f(buffer, dfs[0])
        cached_psdt = cached.psdt
        cached._jax_psd = cached._jax_psdt = None
        calls = _record_psd_preparation(monkeypatch)
        assert backend._prepare_missing_psds_cuda(buffer, dfs)
        assert cached._jax_psd is cached._jax_psdt is None
        first = backend.preload_overwhitened_data_jax(buffer, dfs)
        assert tuple(first) == dfs
        assert len(calls) == 1
        assert calls[0][0][1::2] == dfs[1:]
        assert buffer.psds[dfs[0]] is cached
        assert cached.psdt is cached_psdt
        assert cached._jax_psd is to_jax(cached)
        assert cached._jax_psdt is to_jax(cached_psdt)
        identities = {
            df: (psd, psd.psdt, psd._jax_psd, psd._jax_psdt)
            for df, psd in buffer.psds.items()
        }
        snapshots = {
            df: np.asarray(psd).tobytes() for df, psd in buffer.psds.items()
        }
        assert not backend._prepare_missing_psds_cuda(buffer, dfs)
        assert len(calls) == 1
        second = backend.preload_overwhitened_data_jax(buffer, dfs)
        for df in dfs:
            assert second[df] is first[df]
        buffer.segments.clear()
        backend.preload_overwhitened_data_jax(buffer, dfs)
        assert len(calls) == 1
        for df, previous in identities.items():
            current = buffer.psds[df]
            assert current is previous[0]
            assert current.psdt is previous[1]
            assert current._jax_psd is previous[2]
            assert current._jax_psdt is previous[3]

        replacement = np.asarray(buffer.psd) * np.float32(1.25)
        from pycbc.types import FrequencySeries
        buffer.psd = FrequencySeries(replacement, delta_f=1.0,
                                     epoch=987654321)
        buffer.psds.clear()
        buffer.segments.clear()
        backend.preload_overwhitened_data_jax(buffer, dfs)
        assert len(calls) == 2
        assert calls[-1][0][1::2] == dfs
        reference = _psd_preparation_buffer(psd_values=replacement)
        for df in dfs:
            backend._ensure_psd_for_delta_f(reference, df)
            assert buffer.psds[df] is not identities[df][0]
            assert buffer.psds[df].psdt is not identities[df][1]
            assert np.asarray(buffer.psds[df]).tobytes() != snapshots[df]
            _assert_psd_pair_bytes(buffer.psds[df], reference.psds[df],
                                   ctx.jax_device)


@pytest.mark.parametrize('device', _devices())
@pytest.mark.parametrize('requested_df', [0.25, 0.5])
def test_first_overwhiten_request_prepares_all_required_psd_grids(
        device, requested_df, monkeypatch):
    """The first request is included before the sequential cache lookup."""
    from pycbc.strain import strain_jax as backend

    required = (0.25, 0.125)
    dfs = required if requested_df in required else required + (requested_df,)
    with scheme.JAXScheme(device) as ctx:
        reference = _psd_preparation_buffer()
        expected = {
            df: backend._overwhiten_single_delta_f(reference, df) for df in dfs
        }
        buffer = _psd_preparation_buffer()
        buffer.required_delta_fs = required
        calls = _record_psd_preparation(monkeypatch)
        result = buffer.overwhitened_data(requested_df)
        assert result is buffer.segments[requested_df]
        assert set(buffer.psds) == set(buffer.segments) == set(dfs)
        if device == 'cuda:0':
            assert len(calls) == 1
            assert calls[0][0][1::2] == dfs
        else:
            assert calls == []
        for df in dfs:
            _assert_psd_pair_bytes(buffer.psds[df], reference.psds[df],
                                   ctx.jax_device)
            actual = buffer.overwhitened_data(df)
            assert actual is buffer.segments[df]
            assert actual.start_time == expected[df].start_time
            assert actual.delta_f == expected[df].delta_f
            np.testing.assert_allclose(np.asarray(actual),
                                       np.asarray(expected[df]),
                                       rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize('device', _devices())
@pytest.mark.parametrize('invalid', [
    'empty', 'duplicate', 'zero', 'nonfinite', 'too_long', 'too_short',
    'cutoff', 'source_view', 'source_dtype', 'numpy_scalar', 'list',
    'unhashable',
])
def test_unsupported_psd_batch_does_not_mutate_existing_caches(
        device, invalid, monkeypatch):
    """Rejected batching leaves fallback state and cache identities intact."""
    from pycbc.strain import strain_jax as backend

    with scheme.JAXScheme(device):
        buffer = _psd_preparation_buffer()
        cached = backend._ensure_psd_for_delta_f(buffer, 0.5)
        cached._jax_psd = cached._jax_psdt = None
        cache = buffer.psds
        cached_values = np.asarray(cached).tobytes()
        cached_psdt = cached.psdt
        psdt_values = np.asarray(cached_psdt).tobytes()
        segments = buffer.segments
        segments[0.5] = object()
        segment = segments[0.5]
        requests = {
            'empty': (), 'duplicate': (0.5, 0.25, 0.25),
            'zero': (0.25, 0.0), 'nonfinite': (0.25, np.nan),
            'too_long': (0.25, 0.01), 'too_short': (0.25, 2.0),
            'numpy_scalar': (0.5, np.float32(0.125)),
            'list': [0.5, 0.25], 'unhashable': (0.25, []),
        }.get(invalid, (0.5, 0.25))
        if invalid == 'cutoff':
            buffer.low_frequency_cutoff = 1025
        elif invalid == 'source_view':
            buffer.psd = buffer.psd[:]
        elif invalid == 'source_dtype':
            buffer.psd = buffer.psd.astype(np.complex64)
        elif invalid == 'numpy_scalar':
            buffer.low_frequency_cutoff = np.nextafter(0.125, -np.inf)

        def unexpected_core(*args, **kwargs):
            raise AssertionError('rejected PSD batch dispatched a kernel')

        monkeypatch.setattr(backend, '_multi_psd_prepare_core', unexpected_core)
        assert not backend._prepare_missing_psds_cuda(buffer, requests)
        assert buffer.psds is cache
        assert tuple(cache) == (0.5,)
        assert cache[0.5] is cached
        assert cached.psdt is cached_psdt
        assert cached._jax_psd is cached._jax_psdt is None
        assert np.asarray(cached).tobytes() == cached_values
        assert np.asarray(cached_psdt).tobytes() == psdt_values
        assert buffer.segments is segments
        assert segments == {0.5: segment}


def test_cuda_psd_preparation_leaves_cpu_fallback_available(monkeypatch):
    """Both standard CPU storage and JAX CPU storage retain the old path."""
    from pycbc.strain import strain_jax as backend

    def unexpected_core(*args, **kwargs):
        raise AssertionError('CPU fallback dispatched the CUDA PSD batch')

    monkeypatch.setattr(backend, '_multi_psd_prepare_core', unexpected_core)
    for context in (scheme.CPUScheme(), scheme.JAXScheme('cpu')):
        with context:
            buffer = _psd_preparation_buffer()
            cache = buffer.psds
            segments = buffer.segments
            source = buffer.psd
            assert not backend._prepare_missing_psds_cuda(buffer, (0.5, 0.25))
            assert buffer.psds is cache and cache == {}
            assert buffer.segments is segments and segments == {}
            assert buffer.psd is source
