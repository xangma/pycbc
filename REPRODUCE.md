# Reproducing the measurements

The original commands, working directories, process IDs, return codes and logs are preserved in `live/runs.json`, `smoke/runs.json`, `supplement/runs.json`, `supplement/waveform/manifest.json` and `supplement/inference/process-receipts.json`. Use a **new output directory** for a rerun; do not overwrite this evidence.

The recorded environment is Linux on a Threadripper PRO 3995WX and RTX 4090, Python 3.11.9, Torch 2.13.0+cu130. `packages.json` records the installed distributions; `preparation.json` records imported module locations and rebuilt native-extension hashes. The existing environment supplies LALSuite, NumPy, SciPy and PyCBC's other dependencies. Native extension builds must use the selected interpreter and occur in each exact source checkout.

The original preparation script used a local incremental Git bundle. For a public rerun, obtain the same source commits directly instead:

```sh
BENCH_ROOT=/absolute/path/to/a/new-campaign
BENCH_PY=/absolute/path/to/python
mkdir -p "$BENCH_ROOT"
git clone --no-checkout https://github.com/xangma/pycbc.git "$BENCH_ROOT/main"
git clone --no-checkout https://github.com/xangma/pycbc.git "$BENCH_ROOT/fft"
git clone --no-checkout https://github.com/xangma/pycbc.git "$BENCH_ROOT/cpu"
git -C "$BENCH_ROOT/main" checkout --detach 607bce53ead14f12af32552a5b2441d3bc667267
git -C "$BENCH_ROOT/fft" checkout --detach e6073eaf1a89cfed69af53707f52321eadf129f1
git -C "$BENCH_ROOT/cpu" checkout --detach 1a2ebea088d9e0a31cbb22c19ad24f96ffea2b7c
for BENCH_HEAD in main fft cpu; do
  (cd "$BENCH_ROOT/$BENCH_HEAD" && "$BENCH_PY" setup.py build_ext --inplace -j 4)
done
```

Keep the three source trees clean. Copy the `harness/` directory and original `run-live.py`, `run-smoke.py`, and `run-supplement.py` outside those trees into the new campaign directory. In copies of the runners, set `ROOT` and `PY` to the new campaign root and interpreter. Their recorded affinity is CPUs 8–11; changing affinity, hardware, dependencies or workload changes the experiment and must be recorded. The GPU preflight requires at least 10 GiB free and utilization at most 10%; do not stop unrelated workloads to satisfy it.

Run the three runners sequentially from the new campaign root. `run-smoke.py` waits for live completion and `run-supplement.py` requires successful smoke completion. Inspect scientific statuses as well as process return codes: unsupported capabilities are retained without speed claims. The full campaign contains 126 timed live workers, 18 FFT workers, 20 untimed dispatch probes, 72 supported waveform workers, 90 explicit unsupported waveform records, and 30 inference workers. Smoke results are separate and are not pooled with these measurements.

For individual supplementary commands and timing boundaries, see [FFT harness](harness/fft/README.md), [waveform harness](harness/waveform/README.md), and [inference harness](harness/inference/README.md). The inference `case.json` and `case.npz` must be shared unchanged across every route; they contain the synthetic data, varied parameter points, CPU references and input hashes. FFT cold/warm cache pairs use separate directories; the captured wisdom files and their hashes are included.

Plot generators import the saved data, without importing the measured PyCBC checkout:

```sh
python report/live_report.py --input-dir live --output-dir report
python report/fft_report.py --input-dir supplement/fft --output-dir report
python report/probe_report.py --input-dir supplement/probes --output-dir report
python report/waveform_report.py --input-dir supplement/waveform --output-dir report
python report/inference_report.py --input-dir supplement/inference --output-dir report
```

The generators check the pinned source commits and qualified repetitions, and preserve unsupported or failed cells. They use Matplotlib; the published figures were rendered with the local analysis interpreter, separately from the benchmark interpreter. `sealed-manifest.json` seals the original remote outputs; `SHA256SUMS` additionally covers the published scripts, figures and report text.
