# H1 trigger selection at fixed cuts

This comparison tests whether double working precision in the selected-point
CPU chi-square kernel changes actual single-detector trigger admission.
The data, waveform amplitudes and cuts were not adjusted to create a crossing.
The search uses raw SNR ≥ 5.5 and NewSNR ≥ 5, the same settings as the original
32-template example. These are this search's settings, not universal PyCBC defaults.

| Templates | Candidates before cut | Original retained | Patch retained | Newly retained | Newly rejected |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 89 | 77 | 77 | 0 | 0 |
| 6,144 | 16,378 | 14,334 | 14,335 | 4 | 3 |

The original small example gives a null selection result. Processing the complete
larger bank and all 1,904 valid seconds produces **seven changed decisions**.
Independent complex128 Fourier sums, rounded into float32 event storage before
ranking, agree with the patched decision for all seven. The first changed event
has raw SNR 5.6828446 and NewSNR **4.9986619 → 5.0001907**, with reference
**5.0001830**. Its template and GPS time are in [selection-crossing.json](selection-crossing.json).

This establishes changed single-detector trigger output. It does not establish
a gained/lost detection, changed false-alarm rate or sensitivity. The observed
fraction of changed decisions in this selected bank/data segment is not a rate
estimate for other searches.

## What was held fixed

Both runs used upstream source `454ee900e6faca88a066e71ade489fca9fb221a5` on Linux
on 2026-09-22. The comparison kernel is the arithmetic-only patch
`70effc95e07eb325797d02bc0815eee606ab548d`. The strain is the public H1 frame
`H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf`, with no added signal or amplitude scaling.
Sampling is 2048 Hz, segments are 512 seconds, there are 16 chi-square bins,
and the search FFT backend is MKL. The compared kernel itself performs no FFT.

The 6,144-template bank was already prepared on 2026-09-20, before this
investigation. It is a fixed-seed subset of the
[public O2 bank](https://github.com/gwastro/pycbc-config/blob/710dbfd3590bd93d7679d7822da59fcb6b6fac0f/O2/bank/H1L1-HYPERBANK_SEOBNRv4v2_VARFLOW_THORNE-1163174417-604800.xml.gz),
selected using mass1 ≤ 10, mass2 ≤ 3, chirp mass ≥ 1.2 solar masses and
|aligned spin| ≤ 0.2, with seed 20260919. The 32-template bank is a nested subset.
The included HDF stores the 6,144 physical templates before waveform compression.
Original row selections, input hashes, compression commands, software versions
and population summaries are in [provenance.json](provenance.json).

## How the comparison works

[observe_arithmetic.py](observe_arithmetic.py) records each real
`pycbc_inspiral` selected-point call. It checks exact agreement between the
installed upstream kernel and the archived upstream kernel on every call, and
computes the arithmetic-only result from the same correlation, bins, SNR and
normalization. It always returns the original result to the running search.

At the final cut, it calls the actual `EventManager.newsnr_threshold` twice on
copies of the same clustered events, replacing only chi-square in the second
copy. The event's float32 chi-square storage and int64 degrees-of-freedom array
are preserved. Earlier clustering uses raw SNR; these commands enable no other
chi-square cut or subsequent ranking-based selection. Thus the comparison
measures the trigger output change for this exact command. The original
32-template output also reproduces the earlier saved search's SNR, chi-square,
times and template identities exactly.

The observer's per-call score screen decides which full inputs to save; the
authoritative decisions come from the final public selection method. The screen
and event method have slightly different NumPy division promotion rules.
For these runs, every actual changed event was captured with an independent
reference; this is checked when exporting the population CSVs. The observer is
specific to this command and fixed cut, not a general-purpose search instrument.

The full search used the pinned latest Python sources and existing compiled
extensions only where their Cython source was byte-for-byte unchanged; these
checks and binary hashes are in
[reused-unchanged-extensions.json](reused-unchanged-extensions.json).
Both compared archived kernels were compiled on the Linux host. The notebook's
local kernel replay can differ slightly across compilers and platforms.

## Reproduce

The [notebook](../arithmetic_real_data.ipynb) needs no strain download: it checks
all recorded file hashes, recomputes both cuts for the complete 16,378-candidate
table, and reruns the compiled kernels and independent reference on the first
changed call. Both full candidate tables are included, including unchanged
events. Reference columns are populated for changed events only.

To repeat the full search, install the pinned upstream PyCBC revision above,
the notebook dependencies from the parent README and an MKL FFT backend, then
activate that environment. Download the
[H1 4 kHz GWF file from the public GW170817 noise-subtracted data release](https://gwosc.org/eventapi/json/O1_O2-Preliminary/GW170817/v2/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf)
(the reproducer checks its SHA256). From this directory, run:

```bash
python reproduce.py --frame /path/to/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf \
  --output /path/to/new-empty-output-directory
```

The output directory must not exist. This compresses the included bank, then
runs the observer with the recorded search arguments, writing `selection.json`,
all `point-calls.jsonl`, captured arrays, pre-cut event arrays and original
`triggers.hdf`. Use a fresh directory for each run. The original commands are
in [command-32.json](command-32.json) and [command-6144.json](command-6144.json).
Recompression and numerical results may differ across software/compiler
versions; original hashes and versions are recorded rather than assuming
bitwise reproducibility across environments. The recorded Linux comparison took
about eight minutes and includes observer/reference overhead; it is not a
search-throughput benchmark.
