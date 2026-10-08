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


