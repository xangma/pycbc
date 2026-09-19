.. _jax-parity:

JAX numerical comparison helpers
================================

``tools/jax_parity/`` compares a fixed synthetic corpus between the standard
CPU and JAX execution schemes. It covers array arithmetic, FFT conversion,
frequency shifts, analytical and Welch PSDs, one TaylorF2 waveform, matched
filtering and match interpolation. It does not exercise every module,
chi-squared vetoes, inference likelihoods or complete executable searches.

Comparison policy
-----------------

``policy.json`` is the authoritative set of acceptance targets. The ``jax``
profile uses relative L2 error, with record-specific limits ranging from
``1e-13`` for array arithmetic to ``3e-5`` for the interpolated match value.
Selected records also require elementwise ``allclose`` or exact zero patterns.
The comparator checks record sets, shape, dtype, series metadata and finiteness.
These are targets for this corpus, not a claim that production searches meet
those limits or produce identical triggers.

Complete-search qualification uses the separate frozen gates in
:ref:`jax-reference-campaign`. The fresh 384-template CPU/JAX comparisons
fail those gates; see :ref:`jax-search-qualification` for measured differences.
Passing a helper corpus cannot replace that search-level qualification.

Running the helper corpus
-------------------------

The wrapper generates standard CPU and JAX CPU artifacts in separate
processes using the current checkout, then writes a JSON comparison report:

.. code-block:: console

   CURRENT_PYTHON=python OUTPUT_DIR=artifacts/jax_parity \
     bash tools/jax_parity/run_matrix.sh

Use ``generate.py --help`` to select another available JAX device explicitly.
Keep both generated manifests and NumPy archives together with the report;
the manifests record source, environment and device information.
