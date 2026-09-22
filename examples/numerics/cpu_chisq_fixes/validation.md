# Validation of the CPU chi-square fix drafts

Executed on **2026-09-22**, macOS arm64, Python 3.13.7, NumPy 2.3.5,
SciPy 1.16.3, Cython 3.3.0 and Apple clang 21.0.0. Every variant was installed
editable with PyCBC's own build configuration; explicit `PYTHONPATH` selected
the intended checkout for each test process. The combined source was built in
a separate detached checkout. These results are not Linux/OpenMP or search-scale
performance validation.

| Checkout | Source commit | Focused regressions | Existing `test/test_chisq.py` |
| --- | --- | --- | --- |
| Upstream | `454ee900e6faca88a066e71ade489fca9fb221a5` | Intended precision regressions fail; see [before log](verification/before.txt) | 2 pass |
| π | `2ae4c7301c512fbb15e60321c023ce39d5fdafd3` | 2 pass | 2 pass |
| Time indices | `bffaa25f3175408e5734f6393a13605836c09510` | 2 pass, 2 subtests pass | 2 pass |
| Working arithmetic | `70effc95e07eb325797d02bc0815eee606ab548d` | 5 pass, 4 subtests pass | 2 pass |
| All fixes | Upstream + archived combined source | 9 pass, 6 subtests pass | 2 pass |

The regressions use closed-form answers for phase-sensitive interference,
lost adjacent sample indices, single-coefficient unit power, and zero-phase
addition loss. Compatibility checks cover fractional shifts, input/output dtypes
and an initially populated internal output array. Baseline failures establish
that the accuracy checks detect the original defects. Focused test logs and
legacy-test logs are saved in [verification/](verification/).

Typical commands, with the chosen built checkout at the front of `PYTHONPATH`:

```bash
python -m pytest -q test/test_chisq_cpu_pi.py
python -m pytest -q test/test_chisq_cpu_shifts.py
python -m pytest -q test/test_chisq_cpu_arithmetic.py
python test/test_chisq.py
```

Run each new test file on its corresponding branch, or all three on the combined
checkout. The legacy file is run as a script because it parses command-line
arguments. Editable builds used `uv pip install --no-deps --no-build-isolation
-e CHECKOUT`, with the validation Python selected via `--python`.

Additional checks:

- All five notebooks executed from a fresh kernel without cell errors; HTML
  exports contain their saved outputs. All six plot images were inspected.
  HTML content was checked statically, not visually in a browser.
- Installed and archived kernels agree **exactly** on both input dtypes for all
  three captured samples. The saved JSON records extension paths, source hashes,
  bin powers and the public wrapper's chi-square. Combined results were checked
  against the built combined checkout too.
- Independent direct complex128 sums agree with separate complex128 inverse
  FFTs to a maximum relative difference of approximately `2.22e-16` on this input.
- The three patch files apply cleanly in sequence. Their executable source
  matches the compiled combined snapshot; the latter omits one explanatory
  comment. No fix branch depends on another.
- New regression files and report helpers pass `flake8`. Full
  `flake8 pycbc/ test/` reports **9,503 existing findings** on upstream and each
  fix branch, with no added findings. `git diff --check` passes. `qlty` was not
  installed, so that check was not run.
- The interleaved benchmark covers four layouts, both input dtypes and 1/2/5
  points. It reports medians and interquartile ranges from nine batches; input
  construction is excluded. See [environment](benchmark_environment.json) and
  [all timings](benchmark.csv).

Remaining review questions concern the performance/accuracy tradeoff on search
hosts and compatibility for users calling the internal Cython entry point
directly. The captured replay does not measure population-level detection impact.
