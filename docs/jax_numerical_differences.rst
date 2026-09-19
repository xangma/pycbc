.. _jax-numerical-validation:

JAX numerical differences
=========================

The device implementation is the default. Original-implementation controls
hold selected calculations fixed so that the surrounding JAX calculations can
be validated. They are slower, require concrete inputs and are unavailable to
automatic differentiation. No numerical tolerance is imposed by a control:
it executes the original calculation and preserves its result precision.
Original routes transfer inputs to the CPU and return array results to the active device; these costs are part of the comparison.

Use identical input samples, parameters, grids, library versions and precision
for each comparison. A float32 input already rounded before a calculation
cannot reproduce a float64 input. 

The notebooks assert dtype, shape, values or bytes, and applicable series
metadata against the installed original calculation. Their recorded differences
apply to the displayed inputs, libraries and device. A passing example does not
establish every supported parameter, workload or complete-search decision.

.. toctree::
   :hidden:
   :maxdepth: 1

   jax_array_numerical_differences

Independent calculations
------------------------

Each guide identifies the calculation boundary, explains its difference and
links an executed notebook with fixed inputs and exact original-route checks.

.. list-table:: Calculation guides
   :header-rows: 1
   :widths: 25 75

   * - Calculation
     - Difference and validation
   * - Arrays
     - :doc:`jax_array_numerical_differences`: reductions, cumulative sums,
       elementary functions and selection ties.

Select names through ``JAXScheme(reference_operations=(...))``. Multiple names compose;
select every differing upstream calculation for an exact complete comparison.
Unselected stages remain on their default JAX paths. Whole-stage and finer
controls are documented beside the relevant API.

Interpreting comparisons
-------------------------

Real-arithmetic equivalence does not imply floating-point identity. Changed
operation grouping, a parallel reduction, a different elementary function or
a cancellation-avoiding expression can change rounded values. These changes
can cross a search threshold or change a tied ordering. A default JAX run is
therefore not promised to reproduce every original trigger or sample byte.
The notebooks isolate causes without fitting tolerances to a complete result.
Default numerical mismatches remain failed equality comparisons even when a
higher-precision calculation is closer to an independent mathematical oracle.

An original control is validated against the actual original implementation
at held inputs. Combining controls provides a separate check of the remaining
device implementation, batching and orchestration. Complete search output
comparison, including metadata and explicit exclusions, is described in
Benchmark protocol.
