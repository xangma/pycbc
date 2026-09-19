.. _jax-parity:

JAX Parity Verification
=======================

Numerical parity between JAX and reference NumPy/SciPy implementations is continuously
verified across all modules.

Parity Criteria
---------------

1. **Waveforms**: Overlap match :math:`\mathcal{M} \ge 0.9999` with reference LALSimulation implementations.
2. **Matched Filtering**: Relative difference in maximum SNR :math:`\le 10^{-6}`.
3. **Chi-Squared Vetoes**: Mean relative error :math:`\le 10^{-5}` across power bins.
4. **Likelihood & Inference**: Relative error :math:`\le 10^{-8}` between JAX and CPU log-likelihood ratios.

Running Parity Checks
---------------------

Parity validation tools are located under `tools/jax_parity/`:

.. code-block:: console

   bash tools/jax_parity/run_matrix.sh
