# Persistent full-inspiral campaigns

`run_inspiral_campaign.py` executes the complete `bin/pycbc_inspiral` pipeline
sequentially in one spawned process, including enabled vetoes and HDF trigger
output. It retains imports, the Torch runtime, waveform snapshots and scalar
normalizations between shards. Strain conditioning, PSD construction, filtering,
veto state and the event manager are recreated for each shard.

Use a manifest containing complete `pycbc_inspiral` argument lists:

```json
{"shards": [{"argv": ["--bank-file", "/data/bank.hdf", "..."]}]}
```

Replace `...` with the usual required search arguments. Each shard must specify
`--batch-size` of at least 2 and a Torch CPU or CUDA processing scheme. CPU
processing must use one thread. Use absolute input paths; relative paths resolve
against the runner's working directory. Do not include `--output`, checkpoint
options, multiprocessing options or abbreviated option names.

From the source checkout, in a working PyCBC/Torch environment:

```sh
PYTHONPATH=. python tools/run_inspiral_campaign.py \
  --manifest /data/campaign.json --output /data/new-results --cache-mib 256
```

The output directory must not exist. Outputs are `shard_0.hdf`, `shard_1.hdf`,
etc. An atomically replaced `receipt.json` records progress, worker PIDs,
per-shard arguments, elapsed times and cache counters. Final campaign timing
includes worker startup and shutdown. Failure preserves completed outputs and
writes a failed receipt; inspect the error before retrying into a new directory.

`--fresh-workers` creates one process per shard for comparison. `--cache-mib 0`
disables waveform and normalization retention while retaining process reuse.
The cache budget covers waveform buffer bytes plus estimated per-entry overhead;
it does not cap total process memory or Torch allocator reservations. Choose a
budget large enough for the bank to get reuse; undersized caches can evict a
whole bank before the next shard.

Bank file contents, the thinned parameter table, waveform settings, frequency
geometry and device form the bank key. Changes discard the previous bank's
entries. Normalizations additionally use PSD array content, dtype and frequency
spacing. PSDs are immutable during a shard and newly constructed for each shard.
Cache hits return independent waveform buffers with the current bank metadata.

For scientific comparisons, require exact dataset names, dtypes, shapes and
values under the detector group, plus search intervals. Runtime metadata such
as `templates_per_core` and command lines legitimately differ between runs.
