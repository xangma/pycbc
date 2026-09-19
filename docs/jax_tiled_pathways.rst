.. _jax-tiled-pathways:

JAX Tiled Pathways & Template Partitioning
==========================================

Strategies for executing large template banks exceeding device memory limits.

Partitioning Architecture
-------------------------

Large banks (:math:`N > 10^5`) are partitioned into tiles sized to fit comfortably
within GPU/TPU memory bandwidth and VRAM constraints.

Memory-Constrained Tiling
-------------------------

- Dynamic tile sizing based on template length and duration.
- Overlapped asynchronous tile transfers using double-buffering.
- JIT-compiled search kernel reuse across tiles without recompilation.
