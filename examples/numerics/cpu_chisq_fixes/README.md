# CPU pointwise chi-square: three independent fixes

PyCBC's CPU selected-time chi-square routine rotates a correlation spectrum and
sums power in frequency bins. Three changes address different precision losses.
Each fix branch starts from upstream commit
[`454ee900e`](https://github.com/gwastro/pycbc/commit/454ee900e6faca88a066e71ade489fca9fb221a5),
the upstream `master` fetched on **2026-09-22**. The user's fork `master` was
fast-forwarded to that same commit. Nothing was submitted to `gwastro`.

## Read the evidence

Each notebook introduces the calculation, runs a controlled example against the
actual compiled Cython kernel, and explains its relevance and limits for detector
data. Outputs are saved. HTML exports embed the plots; download/open them in a
browser when GitHub's notebook math rendering is inadequate. MathJax equations
need network access. [Open the HTML index](index.html).

| Question | Executed notebook | HTML | Independent fix |
| --- | --- | --- | --- |
| Does the shortened π constant bias phase? | [π](pi.ipynb) | [Read](pi.html) | [Full-precision π](https://github.com/xangma/pycbc/tree/codex/cpu-chisq-pi) |
| Can float32 change a requested sample index? | [Time indices](time_indices.ipynb) | [Read](time_indices.html) | [Float64 coordinates](https://github.com/xangma/pycbc/tree/codex/cpu-chisq-shifts) |
| Do repeated rotations preserve power? | [Phase](phase.ipynb) | [Read](phase.html) | [Double working arithmetic](https://github.com/xangma/pycbc/tree/codex/cpu-chisq-arithmetic) |
| Can small additions disappear? | [Accumulation](accumulation.ipynb) | [Read](accumulation.html) | Same arithmetic fix; separate controlled evidence |
| What does double working precision change on recorded H1 data? | [Arithmetic replay](arithmetic_real_data.ipynb) | [Read](arithmetic_real_data.html) | Double working arithmetic only |
| How do all three patches interact? | [Combined replay](real_data.ipynb) | [Read](real_data.html) | Supplementary comparison of each patch and their combination |

The arithmetic patch promotes phases, products, within-bin sums and total power
together. The phase and zero-phase accumulation examples isolate mechanisms;
they do **not** assign separate shares of the captured data's error to them.
Input/output storage dtypes remain unchanged. The coordinate patch changes the
internal `point_chisq_code` signature to require float64 shifts; the public
`shift_sum` wrapper converts them automatically.

## Recorded-data result and limits

The input contains three selected samples from a real H1 strain search, using
2048 Hz, 512-second segments, 16 bins and 32 templates. It was selected for a large
discrepancy among six captured calls (11 samples); it is not a prevalence sample.
The [data and capture provenance](../data/point-input-05.json) are included on this
branch. No strain download or new search is needed to replay the selected stage.

On the recorded complex64 input, maximum absolute relative chi-square error falls
from **0.17216%** upstream to **0.0000077%** with all fixes, against a direct
complex128 reference cross-checked with complex128 inverse FFTs. The index fix is
inactive here: every index is exactly representable in float32. Reweighted SNR
changes by up to about **0.00318**, but **none of these candidates crosses the
threshold of 5**. This does not establish sensitivity or false-alarm impact.

Runtime is a review tradeoff. For the captured bin layout and five complex64
points on this Mac, double working arithmetic costs about **1.41×** upstream
kernel time. Ratios vary with shape and point count; the full
[benchmark](benchmark.csv) covers both dtypes and 1/2/5 points.
These are kernel timings, not search-throughput measurements.

## Reproduce

Use Python 3.13 and a C++ compiler. One setup matching the replay's core versions:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'pycbc==2.11.0' 'numpy==2.3.5' 'scipy==1.16.3' \
  'Cython==3.3.0' setuptools wheel matplotlib pandas nbclient nbconvert ipykernel pytest
python -m pip install --no-deps --no-build-isolation \
  'pycbc @ git+https://github.com/gwastro/pycbc.git@454ee900e6faca88a066e71ade489fca9fb221a5'
cd examples/numerics/cpu_chisq_fixes
python execute_notebooks.py
python benchmark.py
```

The release installation supplies dependencies; the following source installation
selects the checked upstream revision. The notebook helper verifies source
SHA256s and builds only the archived kernels requested by each notebook, with
the same compiler flags. Each standalone notebook needs only upstream and its
own patched source; the supplementary combined replay loads all variants.
Build products stay in the system temporary directory.
It does not require switching among fix branches. Linux additionally needs an
OpenMP-capable compiler. The saved run used macOS arm64; Linux performance is not
established by it.

The [source manifest](kernels/manifest.json), exact patches in `kernels/`,
[installed-kernel comparisons](verification/), [validation record](validation.md)
and [benchmark environment](benchmark_environment.json) document the evidence.
Package version metadata in editable builds is less informative than the pinned
source commit. To check another built checkout against its archived counterpart:

```bash
PYTHONPATH=/path/to/built/checkout python verify_installed.py pi
```

Replace `pi` with `upstream`, `shifts`, `arithmetic` or `combined` as appropriate.

## Existing upstream work

The [search record](existing_work.md) covers open and closed issues and PRs.
No exact CPU fix was found. The closest numerical precedent is
[#2950](https://github.com/gwastro/pycbc/pull/2950), which changed CUDA phase
calculation only. The original CPU Cython conversion,
[#2638](https://github.com/gwastro/pycbc/pull/2638), motivates the 1/2/5-point
performance checks. These three fixes can be reviewed and merged independently;
the combined executable source was also built and tested.
