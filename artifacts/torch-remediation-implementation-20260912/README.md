# Torch remediation acquisition, 2026-09-12

This directory publishes the final implementation's finite-fixture evidence.
Read [the implementation report](../../docs/torch_remediation_implementation.md)
for the issue ledger, provider contract and unresolved work;
[measurements.md](measurements.md) gives tables derived from
[summary.json](summary.json). Timings from different boundaries are not
interchangeable.

## Source and host

The tested PyCBC code commit is
`d0aa34d64cc848b00ebff5d3406a65449817f66e` and TorchWave is
`84ef9b3467c8cc34b6d515b967d48646f7297d8f`. The later publication commit
adds documentation and receipts only. [restacked-refs.json](restacked-refs.json)
records the stack before/after integration and the earlier qualification
checkpoint. Branches PR4–PR11 were restacked in order; PR1–PR3 are unchanged.
The original user edits were integrated in their owning branches and also
preserved in the recorded stash. Backup refs remain local. No remote branch
was pushed and no pull request was created or modified.

All numerical tests and final timing/profiler jobs ran on `len`, with cwd
`/home/xangma/pycbc-remediation-20260912/qualified`. The local Mac was used
for code, documentation and Git operations. Runtime Python was
`/home/xangma/pycbc-torch-fixes-20260904-epKdaA/venv/bin/python`;
TorchWave was imported from
`/home/xangma/pycbc-remediation-20260912/torchwave-qualified/src`.

[final-environment.json](final-environment.json) captures OS, CPU, GPU,
driver, runtime packages, filesystem, source cleanliness and Torch build
configuration. CPU: AMD Threadripper PRO 3995WX, 64 physical cores/128
threads. GPU: RTX 4090, 24,564 MiB, driver 610.57.04. Python 3.11.9,
Torch 2.13.0+cu130, NumPy 1.26.4, SciPy 1.13.0, LALSuite 7.21.
TorchWave uses the pinned source import; it has no installed distribution
version in this environment. GPU clocks/power are a post-run snapshot, not
an in-run counter trace. Per-run receipts record thread settings, imported
source snapshots and loaded extension/shared-library hashes.

## Raw archive and verification

[raw-receipts.tar.gz](raw-receipts.tar.gz) includes raw repetitions, final
pytest XML/log, full and F401 lint logs, three offline campaigns and their
HDF outputs, Live output/manifest, synthetic GWF/bank/PSD inputs, the CUDA
trace, profile gates, executable manifests and acquisition scripts.
[raw-receipts-manifest.json](raw-receipts-manifest.json) records the archive
SHA-256 and the size/hash of every archived member. Paths inside the archive
are relative to the target-host acquisition root; source assets and command
paths in receipts preserve their original absolute locations.

Verify and extract into a new directory from this directory:

```sh
python3 - <<'PY'
import hashlib, json, tarfile
from pathlib import Path
m = json.loads(Path('raw-receipts-manifest.json').read_text())
p = Path(m['archive'])
assert hashlib.sha256(p.read_bytes()).hexdigest() == m['archive_sha256']
with tarfile.open(p) as t:
    assert set(t.getnames()) == set(m['files'])
    for name, record in m['files'].items():
        data = t.extractfile(name).read()
        assert len(data) == record['bytes']
        assert hashlib.sha256(data).hexdigest() == record['sha256']
print('Archive and every member verified')
PY
mkdir -p extracted
tar -xzf raw-receipts.tar.gz -C extracted
python3 summarize_results.py --root extracted --prefix final --output regenerated
```

The summarizer refuses failed provider/drain, graph, offline, Live, profile,
veto and test gates, or mixed primary source identities. Its input hash map
binds each derived value to the original receipt. The archive includes a
pinned copy of the old `vetoes.py` solely for the before/after comparison;
the harness verifies old and final module hashes before running.

## Reproduction commands

The exact sequence used on `len` is in [run-final-all.sh](run-final-all.sh),
which serializes GPU work using `flock` through its child scripts. It runs:

1. [run-final-tests.sh](run-final-tests.sh): relevant CPU/CUDA tests and
   repository F401 check, 482 passed and zero skips.
2. [run-final-experiments.sh](run-final-experiments.sh): three actual offline
   campaigns, the explicit-policy survivor comparator, two-rank Live smoke
   fixture, three provider campaigns, graph/500-block engine diagnostics,
   and the independent positive-candidate profiler.
3. [run-final-generation.sh](run-final-generation.sh): six identical-input
   generation experiments (CPU1/CPU4/CUDA, N=2048/131072, B=16), using
   [compare_native_taylorf2_generation.py](compare_native_taylorf2_generation.py).
4. [power-chisq-before-after/benchmark.py](power-chisq-before-after/benchmark.py):
   two warmups and five calls per old/new implementation on clean and
   excluded-prefix-contaminated fixed correlations.

For the original directory layout, launch with:

```sh
cd /home/xangma/pycbc-remediation-20260912
setsid bash run-final-all.sh > logs/final-all.log 2>&1 < /dev/null &
job_pid=$!
printf 'PID/process-group=%s; log=%s\n' "$job_pid" "$PWD/logs/final-all.log"
# Follow progress with: tail -n 30 logs/final-all.log
# Stop this acquisition process group with: kill -- -<recorded-job-pid>
```

The completed final acquisition used PID/process group `3447996`; it exited
0. This is historical identity, not a currently running job or a PID to kill.
All numerical and executable step exits are zero; `final-tests.exit` records
`0 1` for passing tests and the known repository F401 findings, respectively.
[make_cli_fixture.py](make_cli_fixture.py)
and [live_cli_smoke.py](live_cli_smoke.py) construct their deterministic
fixtures. The archive already includes the exact inputs used. On another
host, prepare both pinned repositories and compatible dependencies, update
the explicit root/interpreter/MPI paths in the shell scripts and fixture
manifests, and keep new receipts separate. Do not overwrite the published
acquisition or represent a new environment as the recorded one.

After acquisition, [audit_final_environment.py](audit_final_environment.py)
records environment/lint diagnostics, [summarize_results.py](summarize_results.py)
derives tables, and [archive_receipts.py](archive_receipts.py) produces the
archive. [lint-comparison.json](lint-comparison.json) reports changed-file
F401 success, the unchanged 129 repository-wide F401 diagnostics, non-clean
full style lint, and unavailable `qlty`. Full style lint was not a passing
gate.

## Interpretation and independent review

Provider preparation is B=16, N=2048; each of five fresh workers checks one
first and twenty prepared drains, outside their timers. Generation-only
experiments use prepared banks/arguments and different APIs with explicit
metadata/polarization costs. The offline fixture has four templates and one
survivor per repeated shard. Its nine comparisons use
[offline_pycbc_contract.json](offline_pycbc_contract.json) explicitly;
default comparison mode is not silently relaxed. Live's SNR threshold of
1e6 makes it an empty-trigger dispatch/HDF integration check. The 500-block
graph/stream fixture has zero selected candidates. The separate profiler
has four candidates and two expected accepted injections per cycle.

Independent subagent review inspected the PR4/PR5 code, provider and CLI
receipts, and CUDA trace. No correctness blocker was found in the reviewed
changes. All 21 candidate-array DtoH transfers in the trace occur at public
drain; scalar synchronization remains. Final report claims were checked
against the underlying receipts. This review is advisory and has the finite
coverage described above.

The new veto passes the old cancellation counterexample and lowers measured
allocation, but is slower on the clean comparison. TorchWave's prepared
long-waveform CUDA generation is faster in the sampled APIs; the small
provider campaign shows no cold-start or prepared-filtering gain. Neither
result establishes global optimality, production decision equivalence,
compressed-bank acceleration or optimal whole-workstation use.
