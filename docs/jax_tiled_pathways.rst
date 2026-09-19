.. _jax-tiled-pathways:

JAX template batching
=====================

A bank can contain more templates than fit on the device simultaneously.
The JAX search processes bounded batches selected by ``--batch-size`` and
reuses compiled kernels for compatible shapes. The final partial batch may
have a different shape and require its own compilation.

Choose a batch size for the template length, available memory and FFT
workspace, then validate scientific output and measure the complete search.
The measured batch sweep in :doc:`jax_gpu_investigation` illustrates why a
larger fitting batch need not improve throughput.

Automatic memory-based batch selection and overlapping host preparation with
device execution are possible future experiments. They are not implied by
the current fixed-size batching interface.
