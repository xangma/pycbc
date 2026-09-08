"""Independent precision checks on immutable captured production inputs."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np
from scipy import fft, signal
import scipy

LD = np.longdouble


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error(actual, reference):
    delta = abs(np.asarray(actual, dtype=reference.dtype) - reference)
    relative = delta / np.maximum(abs(reference), np.finfo(LD).tiny)
    return dict(max_absolute=float(delta.max()),
                rms_absolute=float(np.sqrt(np.mean(delta**2))),
                max_relative=float(relative.max()),
                median_relative=float(np.median(relative)),
                p99_relative=float(np.quantile(relative, .99)),
                relative_l2=float(np.sqrt(np.sum(delta**2) / np.sum(abs(reference)**2))))


def reference_psd(samples, metadata, dtype):
    options = metadata['options']
    rate = 1 / metadata['delta_t']
    length = int(options['psd_segment_length'] * rate + .5)
    stride = int(options['psd_segment_stride'] * rate + .5)
    count = options['psd_num_segments']
    fit = (count - 1) * stride + length
    extra = len(samples) - fit
    assert extra >= 0
    values = samples[(extra + 1)//2:len(samples) - extra//2].astype(dtype)
    # Hold the library's float64 Hann coefficients fixed; independently evaluate
    # Welch using SciPy (including its own median-bias implementation).
    window = np.hanning(length).astype(dtype)
    if dtype is LD:
        # signal.welch may allocate complex128 internally even for longdouble
        # input. Use pocketfft directly and assert the transform precision.
        assert options['psd_estimation'] == 'median'
        powers = np.empty((count, length//2+1), dtype=LD)
        for number in range(count):
            spectrum = fft.rfft(values[number*stride:number*stride+length]*window, workers=1)
            assert spectrum.dtype == np.clongdouble
            powers[number] = spectrum.real**2 + spectrum.imag**2
        powers[:, 0] /= 2
        powers[:, -1] /= 2
        k = np.arange(1, (count-1)//2+1, dtype=LD)
        bias = LD(1) + np.sum(1/(2*k+1) - 1/(2*k), dtype=LD)
        estimate = np.median(powers, axis=0) * (LD(2)/(LD(rate)*np.sum(window**2, dtype=LD)*bias))
        frequency = np.arange(length//2+1, dtype=LD)*(LD(rate)/length)
    else:
        frequency, estimate = signal.welch(values, fs=rate, window=window,
            nperseg=length, noverlap=length-stride, detrend=False,
            average=options['psd_estimation'])
    assert estimate.dtype == dtype
    coordinate = np.arange(metadata['length'], dtype=dtype) * dtype(metadata['delta_f']) / dtype(frequency[1])
    left = np.minimum(coordinate.astype(np.int64), len(estimate)-2)
    weight = coordinate - left
    interpolated = estimate[left] * (1-weight) + estimate[left+1] * weight
    size = (len(interpolated)-1)*2
    low = options.get('psd_low_frequency_cutoff') or metadata['low_frequency_cutoff']
    kmin = int(low / metadata['delta_f'])
    inverse = np.zeros(len(interpolated), dtype=dtype)
    assert options['invpsd_trunc_which_spectrum'] == 'invasd'
    assert options.get('invpsd_trunc_low_freq_fill_value', 0) == 0
    inverse[kmin:-1] = interpolated[kmin:-1] ** dtype(-.5)
    impulse = fft.irfft(inverse, n=size, workers=1)
    truncation = int(options['psd_inverse_length'] * rate)
    half = truncation // 2
    assert options['invpsd_trunc_method'] == 'hann'
    taper = np.hanning(truncation).astype(dtype)
    impulse[:half] *= taper[-half:]
    impulse[-half:] *= taper[:half]
    impulse[half:-half] = 0
    with np.errstate(divide='ignore'):
        result = 1 / abs(fft.rfft(impulse, workers=1))**2
    return result, kmin


def direct_chisq(correlation, snr, norm, bins, index, dtype):
    complex_dtype = np.clongdouble if dtype is LD else np.complex128
    pi = np.arccos(dtype(-1))
    sums = []
    for start, stop in zip(bins[:-1], bins[1:]):
        # Exact integer modular phase prevents frequency*time growth from
        # amplifying argument rounding; no recurrence or native kernel is used.
        frequencies = np.arange(start, stop, dtype=np.int64)
        residue = (frequencies * int(index)) % len(correlation)
        phase = residue.astype(dtype) * (2*pi / len(correlation))
        phasor = np.cos(phase).astype(complex_dtype) + 1j*np.sin(phase)
        sums.append(np.sum(correlation[start:stop].astype(complex_dtype)*phasor, dtype=complex_dtype))
    powers = abs(np.asarray(sums))**2
    return (len(sums)*np.sum(powers, dtype=dtype) - abs(complex_dtype(snr))**2) * dtype(norm)**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.capture.resolve(), args.output.resolve()
    out.mkdir()
    metadata = json.loads((root/'capture.json').read_text())
    assert metadata['status'] == 'complete'
    assert np.finfo(LD).nmant > 53
    pins = {str(root/'capture.json'): sha(root/'capture.json')}

    def load(name):
        path = root/name
        expected = metadata['arrays'][name]
        pins[str(path)] = sha(path)
        assert pins[str(path)] == expected['sha256']
        result = np.load(path, allow_pickle=False)
        assert hashlib.sha256(result.tobytes()).hexdigest() == expected['data_sha256']
        return result

    import pycbc
    from pycbc import psd, scheme
    from pycbc.fft.backend_support import set_backend
    from pycbc.filter import make_frequency_series
    from pycbc.types import Array, TimeSeries
    from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed, shift_sum
    result = dict(status='running', numpy=np.__version__, scipy=scipy.__version__,
        python=platform.python_version(), longdouble_mantissa_bits=np.finfo(LD).nmant,
        pycbc_file=pycbc.__file__, script_sha256=sha(__file__),
        scope='fixed captured float32 inputs; numerical accuracy, not search acceptance',
        psd=[], fft=[], point_chisq=[])

    def save():
        (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

    with scheme.CPUScheme(1):
        set_backend(['mkl'])
        strain = load(metadata['strain_file'])
        size = 2*(len(load(metadata['segment_spectra'][0]))-1)
        dt = metadata['strain']['delta_t']
        for number, item in enumerate(metadata['segments']):
            chunk = np.zeros(size, dtype=strain.dtype)
            start, stop = max(item['start'], 0), min(item['stop'], len(strain))
            chunk[start-item['start']:stop-item['start']] = strain[start:stop]
            old = make_frequency_series(TimeSeries(chunk, delta_t=dt)).numpy()
            current = make_frequency_series(TimeSeries(chunk.astype(np.float64), delta_t=dt)).numpy().astype(np.complex64)
            captured = load(metadata['segment_spectra'][number])
            assert np.array_equal(current, captured)
            reference = fft.rfft(chunk.astype(LD), workers=1)*LD(dt)
            ref64 = fft.rfft(chunk.astype(np.float64), workers=1)*dt
            band = slice(int(30*size*dt), -1)
            result['fft'].append(dict(segment=number, capture_exact=True,
                old=error(old[band], reference[band]), current=error(current[band], reference[band]),
                reference64=error(ref64[band], reference[band]),
                current_float32_rounding_exact=bool(np.array_equal(current, reference.astype(np.complex64)))))
            print('FFT', number, flush=True)
            save()

        # Replay the old public from_cli source using proposed globals after
        # verifying that the sole promotion statement is removed. This isolates
        # just PSD precision and cannot mutate the imported function binding.
        source = inspect.getsource(psd.from_cli)
        target = '            strain = strain.astype(float64)\n'
        assert source.count(target) == 1
        namespace = dict(psd.from_cli.__globals__)
        exec(compile(source.replace(target, '            strain = strain\n'), '<isolated-old-PSD>', 'exec'), namespace)
        legacy_psd = namespace['from_cli']
        for number, item in enumerate(metadata['psd_calls']):
            samples = load(item['input_file'])
            options = SimpleNamespace(**item['options'])
            options.asd_file = None
            series = TimeSeries(samples, delta_t=item['delta_t'])
            argv = (options, item['length'], item['delta_f'], item['low_frequency_cutoff'])
            old = legacy_psd(*argv, strain=series, precision='single').numpy()
            current = psd.from_cli(*argv, strain=series, precision='single').numpy()
            captured = load(item['output_file'])
            assert np.array_equal(current, captured)
            assert np.array_equal(old, load('psd-original.npy'))
            ref, kmin = reference_psd(samples, item, LD)
            ref64, _ = reference_psd(samples, item, np.float64)
            band = slice(kmin, -1)
            result['psd'].append(dict(call=number, current_capture_exact=True, old_frozen_exact=True,
                hann_coefficients='shared float64 np.hanning coefficients; arithmetic evaluated independently',
                old=error(old[band], ref[band]), current=error(current[band], ref[band]),
                reference64=error(ref64[band], ref[band]),
                old_closer_frequencies=int(np.count_nonzero(abs(old[band]-ref[band]) < abs(current[band]-ref[band]))),
                current_closer_frequencies=int(np.count_nonzero(abs(current[band]-ref[band]) < abs(old[band]-ref[band])))))
            print('PSD', number, flush=True)
            save()

        known = {(370,4):1069877, (107,0):1601270, (283,0):1654835, (348,0):1654835}
        for item in metadata['points']:
            correlation, snrv, indices, observed = (load(item[name]) for name in
                ('correlation_file', 'snr_file', 'indices_file', 'chisq_file'))
            if not len(indices):
                continue
            corr = Array(correlation)
            bins, norm = np.asarray(item['bins']), item['norm']
            current = power_chisq_at_points_from_precomputed(corr, snrv, norm, bins, indices)
            assert np.array_equal(current, observed)
            old = (shift_sum(corr, indices, bins)*(len(bins)-1) - (snrv.conj()*snrv).real)*norm**2
            selected = {0, int(np.argmax(abs(current-old)))}
            target = known.get((item['index'], item['segment']))
            selected.update(np.flatnonzero(indices == target).tolist())
            for position in sorted(selected):
                reference = direct_chisq(correlation, snrv[position], norm, bins, indices[position], LD)
                ref64 = direct_chisq(correlation, snrv[position], norm, bins, indices[position], np.float64)
                result['point_chisq'].append(dict(template=item['index'], segment=item['segment'],
                    sample=int(indices[position]), capture_exact=True, old=float(old[position]),
                    current=float(current[position]), reference_decimal=str(reference),
                    old_absolute_error=float(abs(old[position]-reference)),
                    current_absolute_error=float(abs(current[position]-reference)),
                    reference64_absolute_error=float(abs(ref64-reference)),
                    normalization=norm))
            print('point', item['index'], item['segment'], flush=True)
            save()
    result.update(status='complete', input_sha256=pins,
        inputs_unchanged=all(sha(path) == digest for path, digest in pins.items()))
    assert result['inputs_unchanged']
    save()


if __name__ == '__main__':
    main()
