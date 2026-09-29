"""JAX conditioning filters with the original PyCBC/LAL conventions."""

import functools
import jax
import jax.numpy as jnp

from pycbc.types import TimeSeries


def _compensated_pairwise_sum(values):
    """Sum a fixed tree, retaining each pair's float64 rounding residual.

    A generic GPU reduction can use different trees across fresh processes.
    For FIR normalization that changed one block's float32 conditioned input.
    The explicit tree keeps the sum reproducible without a host transfer.
    """
    length = len(values)
    padded_length = 1 << (length - 1).bit_length()
    totals = jnp.pad(values, (0, padded_length - length))
    residuals = jnp.zeros_like(totals)
    while padded_length > 1:
        pairs = totals.reshape((padded_length // 2, 2))
        left, right = pairs[:, 0], pairs[:, 1]
        total = left + right
        right_virtual = total - left
        rounding_error = ((left - (total - right_virtual))
                          + (right - right_virtual))
        old_residuals = residuals.reshape((padded_length // 2, 2))
        residuals = (old_residuals[:, 0] + old_residuals[:, 1]) + rounding_error
        totals = total
        padded_length //= 2
    return totals[0] + residuals[0]


@functools.partial(jax.jit, static_argnames=("numtaps", "window", "pass_zero"))
def firwin(numtaps, cutoff, window=('kaiser', 5.0), pass_zero=True):
    """Windowed-sinc FIR design used by PyCBC's Kaiser filters."""
    cutoffs = jnp.atleast_1d(jnp.asarray(cutoff, dtype=jnp.float64))
    ends_at_nyquist = bool(len(cutoffs) % 2) != pass_zero
    edges = jnp.concatenate((jnp.zeros(1) if pass_zero else jnp.empty(0),
                             cutoffs,
                             jnp.ones(1) if ends_at_nyquist else jnp.empty(0)))
    bands = edges.reshape((-1, 2))
    m = jnp.arange(numtaps, dtype=jnp.float64) - (numtaps - 1) / 2.0
    h = jnp.sum(bands[:, 1, None] * jnp.sinc(bands[:, 1, None] * m)
                - bands[:, 0, None] * jnp.sinc(bands[:, 0, None] * m), axis=0)
    if window[0] != 'kaiser':
        raise NotImplementedError("JAX FIR design currently supports Kaiser windows")
    h *= jnp.kaiser(numtaps, window[1])
    left, right = bands[0]
    scale_frequency = jnp.where(left == 0, 0.0,
                               jnp.where(right == 1, 1.0, (left + right) / 2))
    terms = h * jnp.cos(jnp.pi * m * scale_frequency)
    return h / _compensated_pairwise_sum(terms)


def _raw(x):
    data = getattr(x, "_data", x)
    return getattr(data, "array", data)


def _series(values, source, delta_t=None):
    return TimeSeries(values, epoch=source.start_time,
                      delta_t=source.delta_t if delta_t is None else delta_t,
                      dtype=source.dtype)


@jax.jit
def _circular_fir(x, b):
    """Apply the native PyCBC circular FIR convention.

    The CPU implementation FFTs ``reverse(b)`` after the same roll used by
    ``Array.resize``/``Array.roll``, then computes ``conj(C) * X``.  This is
    algebraically the same circular convolution as ``X * FFT(b)``, but keeps
    the native coefficient placement and correlation operation order.  The
    explicit scale pair mirrors the native unnormalised inverse FFT followed
    by division by ``N``; it is deliberately not a claim of bitwise FFT
    equivalence across backends.
    """
    n = len(x)
    fillen = len(b)
    reversed_coefficients = jnp.pad(b[::-1], (0, n - fillen))
    rolled_coefficients = jnp.roll(reversed_coefficients, n - fillen + 1)
    cfreq = jnp.fft.rfft(rolled_coefficients)
    tfreq = jnp.fft.rfft(x)
    spectrum = jnp.conj(cfreq) * tfreq
    # On some CUDA/JAX versions, real inverse FFT normalization for
    # non-power-of-two float64 lengths is performed with a lower-precision
    # reciprocal internally.  Reconstructing the Hermitian spectrum and
    # using the complex FFT avoids that backend-specific scale error.  Keep
    # the cheaper real transform for float32 and power-of-two lengths.
    is_power_of_two = n > 0 and (n & (n - 1)) == 0
    if x.dtype == jnp.float64 and not is_power_of_two:
        tail = spectrum[-2:0:-1] if n % 2 == 0 else spectrum[:0:-1]
        full_spectrum = jnp.concatenate((spectrum, jnp.conj(tail)))
        inverse = jnp.fft.fft(jnp.conj(full_spectrum)).real
    else:
        # ``norm='forward'`` makes this inverse unnormalised, matching PyCBC's
        # FFT backend; retain the native explicit division boundary afterward.
        inverse = jnp.fft.irfft(spectrum, n=n, norm="forward")
    return inverse / jnp.asarray(n, dtype=x.dtype)


@jax.jit
def _circular_fir_zero(values, coefficients):
    """Apply the circular FIR path and its standard zero/shift correction."""
    values = jnp.asarray(values)
    coefficients = jnp.asarray(coefficients, dtype=values.dtype)
    filtered = _circular_fir(values, coefficients)
    nzero = (len(coefficients) // 2) * 2
    filtered = filtered.at[:nzero].set(0)
    return jnp.roll(filtered, -len(coefficients) // 2)


def lfilter(coefficients, timeseries):
    """Preserve PyCBC's short causal / long circular FIR convention."""
    x = jnp.asarray(_raw(timeseries))
    if len(x) < 128:
        # The native scipy.signal.lfilter call uses denominator 1.0,
        # promoting real/complex single-precision inputs to float64/complex128.
        # Preserve that precision without transferring samples to the host.
        compute_dtype = jnp.result_type(x.dtype, jnp.asarray(coefficients).dtype,
                                        jnp.float64)
        b = jnp.asarray(coefficients, dtype=compute_dtype)
        values = jnp.convolve(x.astype(compute_dtype), b, mode='full')[:len(x)]
        return TimeSeries(values, epoch=timeseries.start_time,
                          delta_t=timeseries.delta_t)
    b = jnp.asarray(coefficients, dtype=x.dtype)
    return _series(_circular_fir(x, b), timeseries)


def fir_zero_filter(coefficients, timeseries):
    out = lfilter(coefficients, timeseries)
    n = (len(coefficients) // 2) * 2
    values = jnp.roll(jnp.concatenate((jnp.zeros((n,), dtype=out.dtype),
                                       jnp.asarray(_raw(out))[n:])),
                      -len(coefficients) // 2)
    return _series(values, timeseries)


@functools.partial(jax.jit, static_argnames=("order", "highpass"))
def _butterworth_core(values, frequency, delta_t, attenuation, order, highpass):
    # LAL's amplitude is the squared forward/reverse response.
    wc = jnp.tan(jnp.pi * frequency * delta_t)
    exponent = (0.5 if highpass else -0.5) / order
    wc *= (1.0 / jnp.sqrt(1.0 - attenuation) - 1.0) ** exponent
    dtype = values.dtype

    def section(x, b, a):
        # Direct form II state is [w[n], w[n-1]]. Each sample applies an
        # affine map to the preceding state. Compose those maps in a parallel
        # prefix scan to avoid one GPU iteration per sample. Histories remain
        # double even for REAL4 data; only each pass's output is cast, as in
        # XLALIIRFilterVector. Tree composition changes rounding, not the
        # recurrence, coefficients, initial state or section/pass ordering.
        transition = jnp.array([[-a[0], -a[1]], [1.0, 0.0]])

        def compose(left, right):
            left_a, left_b = left
            right_a, right_b = right
            return (right_a @ left_a,
                    (right_a @ left_b[..., None])[..., 0] + right_b)

        def apply(samples):
            transitions = jnp.broadcast_to(transition, (len(samples), 2, 2))
            offsets = jnp.stack((samples.astype(jnp.float64),
                                 jnp.zeros(len(samples), jnp.float64)), axis=-1)
            _, states = jax.lax.associative_scan(compose,
                                                (transitions, offsets))
            two_back = jnp.concatenate((jnp.zeros(1, jnp.float64),
                                        states[:-1, 1]))
            return (b[0] * states[:, 0] + b[1] * states[:, 1]
                    + b[2] * two_back).astype(dtype)

        forward = apply(x)
        reverse = apply(forward[::-1])
        return reverse[::-1]

    # Preserve LAL's pole-pair order: weakest damping first. Reordering SOS
    # changes intermediate rounding when storing single-precision samples.
    def pair(i, samples):
        damping = 2.0 * wc * jnp.sin(jnp.pi * (i + 0.5) / order)
        denom = 1.0 + damping + wc * wc
        b = (jnp.array([1.0, -2.0, 1.0]) if highpass else
             wc * wc * jnp.array([1.0, 2.0, 1.0])) / denom
        a = jnp.array([2.0 * (wc * wc - 1.0),
                       1.0 - damping + wc * wc]) / denom
        return section(samples, b, a)

    # Compile the prefix graph once and reuse it for each pole pair.
    values = jax.lax.fori_loop(0, order // 2, pair, values)
    if order % 2:
        denom = 1.0 + wc
        b = (jnp.array([1.0, -1.0, 0.0]) if highpass else
             wc * jnp.array([1.0, 1.0, 0.0])) / denom
        a = jnp.array([(wc - 1.0) / denom, 0.0])
        values = section(values, b, a)
    return values


@functools.partial(jax.jit, static_argnames=("order",))
def _highpass_lal_serial_core(values, frequency, delta_t, attenuation, order):
    """Follow LAL's high-pass section coefficients on the active JAX device.

    Direct Form II state is [w[n], w[n-1]]. Each sample applies an
    affine map to the preceding state. Compose those maps in a parallel
    prefix scan (jax.lax.associative_scan) to execute in parallel on the
    device without host transfers or serial recurrence.
    """
    wc = jnp.tan(jnp.pi * frequency * delta_t)
    wc *= (1.0 / jnp.sqrt(1.0 - attenuation) - 1.0) ** (0.5 / order)

    def compose(left, right):
        left_a, left_b = left
        right_a, right_b = right
        return (right_a @ left_a,
                (right_a @ left_b[..., None])[..., 0] + right_b)

    def section(samples, b, r):
        transition = jnp.array([[r[1], r[2]], [1.0, 0.0]], dtype=jnp.float64)

        def apply_pass(x):
            n = len(x)
            transitions = jnp.broadcast_to(transition, (n, 2, 2))
            offsets = jnp.stack((x.astype(jnp.float64),
                                 jnp.zeros(n, jnp.float64)), axis=-1)
            _, states = jax.lax.associative_scan(compose, (transitions, offsets))
            two_back = jnp.concatenate((jnp.zeros(1, jnp.float64), states[:-1, 1]))
            return (b[0] * states[:, 0] + b[1] * states[:, 1]
                    + b[2] * two_back).astype(values.dtype)

        forward = apply_pass(samples)
        reverse = apply_pass(forward[::-1])
        return reverse[::-1].astype(values.dtype)

    def pair(i, samples):
        theta = jnp.pi * (i + 0.5) / order
        real = wc * jnp.cos(theta)
        imag = wc * jnp.sin(theta)
        ratio = -real / (1.0 + imag)
        denom = 1.0 + imag - ratio * real
        pole_real = (1.0 - imag + ratio * real) / denom
        pole_imag = (real - ratio * (1.0 - imag)) / denom
        gain_real = ratio / denom
        gain_imag = 1.0 / denom
        zero_a = gain_real + 1j * gain_imag
        zero_b = -gain_real + 1j * gain_imag
        gain = ((1.0 + 0j) * (zero_a * (-1j)) *
                (zero_b * (-1j))).real
        b = jnp.stack((gain, -2.0 * gain, gain))
        r = jnp.stack((-jnp.ones_like(pole_real), 2.0 * pole_real,
                       -(pole_real * pole_real + pole_imag * pole_imag)))
        return section(samples, b, r)

    for i in range(order // 2):
        values = pair(i, values)
    if order % 2:
        denom = 1.0 + wc
        gain = 1.0 / denom
        b = jnp.stack((gain, -gain, jnp.float64(0)))
        r = jnp.stack((jnp.float64(-1), (1.0 - wc) / denom,
                       jnp.float64(0)))
        values = section(values, b, r)
    return values


def butterworth(timeseries, frequency, order=8, attenuation=0.1,
                btype="lowpass"):
    if not 0 < frequency < 0.5 / timeseries.delta_t:
        raise ValueError("frequency must be between zero and Nyquist")
    if order <= 0 or not 0 < attenuation < 1:
        raise ValueError("positive order and attenuation between zero and one required")
    from pycbc import scheme
    mode = getattr(scheme.mgr.state, 'jax_highpass_mode', 'lal-serial')
    if mode == 'lal-serial':
        if btype == 'highpass':
            if not jax.config.jax_enable_x64:
                raise RuntimeError("lal-serial high-pass requires JAX 64-bit support")
            values = _highpass_lal_serial_core(jnp.asarray(_raw(timeseries)),
                                               frequency, timeseries.delta_t,
                                               attenuation, order)
        else:
            values = _butterworth_core(jnp.asarray(_raw(timeseries)), frequency,
                                       timeseries.delta_t, attenuation, order,
                                       False)
    else:
        values = _butterworth_core(jnp.asarray(_raw(timeseries)), frequency,
                                   timeseries.delta_t, attenuation, order,
                                   btype == 'highpass')
    return _series(values, timeseries)


def resample_ldas(timeseries, delta_t, coefficients, factor):
    filtered = fir_zero_filter(coefficients, timeseries)
    values = jnp.asarray(_raw(filtered))[::factor]
    out = TimeSeries(values, epoch=timeseries.start_time, delta_t=delta_t,
                     dtype=timeseries.dtype)
    out.corrupted_samples = 10
    return out


def resample_to_delta_t(timeseries, delta_t, method):
    """Dispatch supported resampling without a native-backend fallback."""
    factor = int(round(delta_t / timeseries.delta_t))
    expected = factor * timeseries.delta_t
    if factor < 1 or abs(delta_t - expected) > 1e-8 + 1e-5 * abs(expected):
        raise ValueError("JAX resampling requires an integer sample-rate factor")
    if method == 'ldas':
        coefficients = firwin(factor * 20 + 1, 1.0 / factor,
                              window=('kaiser', 5))
        return resample_ldas(timeseries, delta_t, coefficients, factor)
    if method == 'butterworth':
        raise NotImplementedError(
            "JAX Butterworth resampling is not yet implemented; use method='ldas'")
    raise ValueError("Unsupported JAX resampling method: %s" % method)
