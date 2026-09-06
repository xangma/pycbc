"""Bind the before/after CPU profile attribution to raw pstats and source blobs."""
import hashlib
import json
from pathlib import Path
import pstats
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = 'a4d77a6d1863c0515e8dace64c5609b63d40b51e'
BASELINE = '837f38d493420043e45fb1ad210a0ccf68bacbaa'
REPO = '/private/tmp/pycbc-torch-inspiral-hotpaths-20260906'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    paths = {
        'before_cpu': 'runs/reference-precision5-cpu-l512-cprofile/profile.pstats',
        'before_torch_cpu': 'runs/profile-precision5-torch-cpu-l512-cprofile/profile.pstats',
        'after_cpu': 'runs/profile-optimized6-cpu-l512-cprofile/profile.pstats',
        'after_torch_cpu': 'runs/profile-optimized6-torch-cpu-l512-cprofile/profile.pstats',
    }
    inputs = {name: sha((ROOT / name).read_bytes()) for name in [*paths.values(),
        'optimized-report-v6/report.json', 'final-report/report.json', Path(__file__).name]}
    profiles = {}
    for role, name in paths.items():
        stats = pstats.Stats(str(ROOT / name))

        def select(suffix, function):
            matches = [(key, value) for key, value in stats.stats.items()
                       if key[0].endswith(suffix) and key[2] == function]
            if suffix == 'fft/torchfft.py':
                matches = sorted(matches, key=lambda item: item[1][3], reverse=True)[:1]
            if len(matches) != 1:
                raise ValueError((role, suffix, function, len(matches)))
            key, (primitive, calls, self_seconds, cumulative, _) = matches[0]
            assert calls == (480 if function == 'execute' else 96), (role, key, calls)
            return dict(file=key[0], line=key[1], function=key[2], primitive_calls=primitive,
                        calls=calls, self_seconds=self_seconds, cumulative_seconds=cumulative)

        torch = 'torch' in role
        profiles[role] = dict(total_self_seconds=stats.total_tt,
            interpolation=select('waveform/decompress_torch.py' if torch else
                                 'waveform/decompress_cpu.py', 'inline_linear_interp'),
            inverse_fft=select('fft/torchfft.py' if torch else 'fft/mkl.py', 'execute'))
    # The outer IFFT wrapper has the largest cumulative execute time; native
    # workspace calls are nested inside it and must not be counted again.
    before, after = profiles['before_torch_cpu'], profiles['after_torch_cpu']
    normal = profiles['after_cpu']
    saved = {key: before[key]['cumulative_seconds'] - after[key]['cumulative_seconds']
             for key in ('interpolation', 'inverse_fft')}
    total_saved = before['total_self_seconds'] - after['total_self_seconds']
    remaining = after['total_self_seconds'] - normal['total_self_seconds']
    fft_gap = after['inverse_fft']['cumulative_seconds'] - normal['inverse_fft']['cumulative_seconds']
    blobs = {}
    for commit in (BASELINE, SOURCE):
        blobs[commit] = {name: sha(subprocess.check_output(['git', '-C', REPO, 'show', commit + ':' + name]))
            for name in ('pycbc/fft/torchfft.py', 'pycbc/fft/mkl.py',
                         'pycbc/waveform/decompress_torch.py', 'pycbc/waveform/decompress_cpu.py')}
    result = dict(schema_version=1, source_commit=SOURCE, baseline_source_commit=BASELINE,
        input_sha256=inputs, source_blob_sha256=blobs, profiles=profiles,
        saved_cumulative_seconds=saved, profile_total_reduction_seconds=total_saved,
        reduction_explained_fraction=sum(saved.values()) / total_saved,
        remaining_profile_gap_seconds=remaining, remaining_ifft_gap_seconds=fft_gap,
        remaining_gap_ifft_fraction=fft_gap / remaining,
        limitations=['Separate instrumented processes; no confidence intervals or timing samples.',
                     'Only the two disjoint call paths are added; nested cumulative times are not.',
                     'Retained FP64 MKL FFT and conversions are not individually isolated.'])
    prose = f'''# Optimized CPU hot-path profiles

Source `{SOURCE}` is compared with `{BASELINE}` using separate cProfile runs
of the selected 96-template, 512-second, one-thread workload. The accompanying
[JSON](profile-interpretation-v6.json) binds four raw pstats files, both reports
and eight source blobs. All numbers below are instrumented cumulative seconds.

| Call path | Torch CPU before | Torch CPU after | Normal CPU after |
|---|---:|---:|---:|
| Compressed interpolation (96 calls) | {before['interpolation']['cumulative_seconds']:.3f} | {after['interpolation']['cumulative_seconds']:.3f} | {normal['interpolation']['cumulative_seconds']:.3f} |
| Inverse FFT (480 calls) | {before['inverse_fft']['cumulative_seconds']:.3f} | {after['inverse_fft']['cumulative_seconds']:.3f} | {normal['inverse_fft']['cumulative_seconds']:.3f} |

Torch CPU profile total fell from {before['total_self_seconds']:.3f} to
{after['total_self_seconds']:.3f} seconds. These disjoint paths account for
{sum(saved.values()):.3f} seconds, or {100 * sum(saved.values()) / total_saved:.2f}%
of that reduction. The existing compiled interpolation routine is now reached
through shared CPU array views. The large search inverse FFT uses retained
double-precision MKL workspaces with input promotion and output conversion.
Both paths retain their qualification and fallback constraints.

The after-profile gap to normal CPU is {remaining:.3f} seconds. The inverse-FFT
gap is {fft_gap:.3f} seconds ({100 * fft_gap / remaining:.2f}% of the total gap).
Normal CPU uses single-precision MKL; Torch's qualified large transform retains
double precision. Precision, conversion and library/planning effects have not
been isolated individually, so this comparison does not assign a separate cost
to each. Nested native-plan and copy times must not be added to wrapper time.

For capacity, use the nine separate unprofiled processes in
[the optimized report](optimized-report-v6/report.json). Native perf cycle shares
and CUDA event durations use different denominators and cannot be added to these
Python profile seconds. See [REPRODUCE.md](REPRODUCE.md) to restore raw profiles.
'''
    for name, data in (('profile-interpretation-v6.json', json.dumps(result, indent=2) + '\n'),
                       ('profile-interpretation-v6.md', prose)):
        with (ROOT / name).open('x') as stream:
            stream.write(data)
    print(json.dumps({key: result[key] for key in ('reduction_explained_fraction',
          'remaining_gap_ifft_fraction')}))


if __name__ == '__main__':
    main()
