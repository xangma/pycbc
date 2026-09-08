# CPU correction runtime cost

Four trials per arm on len, CPU 8 (SMT sibling 72), one thread per runtime, MKL; a shared host with no reservation. Runs alternated original/corrected order in each pair. Frozen inputs comprise 384 compressed templates, five segments and 1,904 valid seconds.

| Source | Wall seconds, all trials | Median seconds |
| --- | --- | ---: |
| original | 65.27930, 65.47842, 65.27659, 65.28078 | 65.28004 |
| corrected | 67.98432, 68.03117, 68.09225, 68.13230 | 68.06171 |

Corrected/original median wall ratio is 1.04261137, a **4.26% increase** (2.78 seconds). This describes this finite workload and shared-host sample, without a confidence or sustained-capacity claim. It measures the cost of changed arithmetic and changed output; it is not an equal-output speedup comparison.

Each timing trial passed the frozen comparator against its own source's qualified output. Original has 1,988 triggers and corrected has 1,991. No numerical budget was relaxed. All tracked source files, native binaries, generated version files, harness files and bank/frame hashes were unchanged across the experiment; no Torch was imported at the three checked runtime snapshots.

Timing starts at fresh checked-inspiral process launch and ends after executable exit, HDF writing and runtime/native verification. Qualification is excluded. Per-run /usr/bin/time results and host observations were retained, including CPU 8 and SMT sibling 72. Host monitoring observes contention but does not reserve resources.

[Full summary](linux-acquisition/cpu-cost-v2/summary.json) · [Driver](run-cpu-cost-v2.py) · [All source/input/harness pins](linux-acquisition/cpu-cost-v2/pins.json)

The first cost controller stopped before running any trial because the standalone qualification used a different receipt layout. A comparator receipt was then derived from the completed, pinned qualification records; its provenance names the source records. The raw first attempt is retained separately. No timing measurement was fabricated or reused.
