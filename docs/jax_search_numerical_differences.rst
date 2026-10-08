.. _jax-search-numerical-differences:

JAX search numerical differences
================================

Batching preserves the ordinary frequency bands, template duration groups and
PSD grids. Floating-point operations, scalar promotion, storage casts and
sort ordering can still differ. A small value difference can cross a threshold;
equal values can select different tied event indices.

The :download:`search comparison notebook
<../examples/jax/jax_search_numerical_differences.ipynb>` uses deterministic
synthetic inputs and the original CPU implementations as oracles. It records
library versions and device hardware, compares default results, and asserts
equal shape, dtype and bytes with selected original operations. Its component
examples do not establish complete-search equivalence. See :doc:`jax_search`
for interfaces and the selector table.

Batched filtering and normalization
------------------------------------

The batched filter forms ``conj(template) * data`` before PSD division, then
performs an inverse transform and SNR normalization. Its cutoff bins follow
the single-template API. Complex output storage retains the input precision;
inner-product accumulation uses double precision. Correlation arithmetic,
complex division, FFT algorithms and reduction grouping have separate
floating-point boundaries. Output comparisons alone cannot identify the
precise internal FFT operation responsible for a bin difference.

The notebook composes ``correlate``, ``divide``, ``ifft`` and ``weighted_inner``
to reproduce the original PSD-weighted SNR and normalization bytes for
complex64 and complex128 inputs. Without a PSD, use ``inner`` instead of
``weighted_inner``. Other JAX code, including batch assembly and cutoff handling,
remains active in these comparisons.

Generic Live normalization follows ``sigma_cached``: float32 complex power
and inverse-PSD weighting, with double-precision accumulation over each
template's original frequency support. ``squared_norm``, ``divide`` and
``inner`` independently select the original parts. Precomputed waveform
normalizations retain their separate endpoint-subtraction contract.

Live thresholds and storage
----------------------------

The generic original normalization reduction returns a Python float. NumPy
treats that scalar weakly when multiplying a float32 magnitude or a complex64
array. JAX's batched normalization is a strong float64 array: multiplication
occurs in wider precision, followed by a cast when storing complex64. Matching
the stored dtype therefore does not guarantee matching the rounded product.

For the notebook's fixed complex64 peak
``-0.17656055 + 0.93697107j`` and normalization ``2.7859016728226313``, the
original magnitude product rounds to ``2.656249523162842``. Setting the
threshold to that value accepts the original result, while the wider product
falls below it. The stored scaled real part is ``-0.49188036`` in the original
calculation and ``-0.49188033`` after wider multiplication and casting.
The notebook holds peak and normalization fixed to isolate this mechanism.

``live_selection`` executes the original Live scaling, threshold and result
storage with the supplied peaks and normalizations. It preserves whether the
original normalization was a weak Python float or a NumPy scalar. Combine it
with ``abs_arg_max`` when validating the original peak-selection calculation.
The notebook also uses the public Live processing API to assert every result
column and the correlation, peak, normalization and sample index passed to
the power chi-square veto. That composition includes the original filter,
selection, chi-square bin and point-calculation controls.

Ranking and event selection
-----------------------------

Both ranking backends promote ``newsnr`` and ``effsnr`` inputs to float64.
Powers, divisions and compilation remain possible numerical boundaries;
agreement on one fixture does not prove disagreement elsewhere. The notebook
reports measured equality or differences and checks each original ranking
control, without requiring a default mismatch.

``event_chisq_threshold``, ``event_newsnr_threshold`` and ``event_loudest``
execute the respective original event-manager selections. These preserve
field casts, comparisons and equal-score ordering. They hold the existing
event columns fixed, so upstream filtering can be validated independently.

Segments, clustering and coincidence geometry
----------------------------------------------

JAX's stable time ordering can differ from the original NumPy default
quicksort for equal keys. Segment membership can match while returned index
order differs. For coincidence clustering, equal times and equal statistics
can select different tied input indices. The notebook demonstrates both cases
and restores original ordering with ``segment_veto``, ``time_coincidence``,
``cluster_over_time``, ``cluster_coincs`` or ``cluster_coincs_multiifo`` as
appropriate. FindChirp uses the original int32 time conversion before greedy
comparisons; its example checks that contract and ``findchirp_cluster``.

The original two-detector coincidence path forms an absolute mean in float64,
then converts to NumPy ``longdouble`` for slide separation and clustering.
JAX forms a relative mean after subtracting an anchor and uses float64.
These different sequences place rounding differently. At GPS time
``1e9``, averaging times separated by one float64 ULP rounds away a half-ULP
offset in the original absolute mean; the relative calculation retains it.
With a second candidate 0.25 seconds later and a 0.25-second window, the
notebook's original path retains indices ``[0, 1]`` and JAX retains ``[1]``.
The ``cluster_coincs`` control restores the complete original mean/slide/
clustering stage. The multi-detector control likewise restores its original
participating-detector mean and geometry.

Quadrature arithmetic
-----------------------

The original quadrature statistic spells squares and square roots as powers
with scalar exponents ``2.`` and ``0.5``. NumPy's
`power loop <https://github.com/numpy/numpy/blob/v2.5.2/numpy/_core/src/umath/loops_umath_fp.dispatch.c.src#L139-L163>`_
specializes those scalar exponents to multiplication and square root. JAX's
default quadrature implementation uses multiplication and square root
explicitly, keeping the calculation on device.

The notebook separately compares eager floating powers, integer powers,
multiplication and JIT compilation on identical float32 inputs. In its recorded
CPU execution, three of 100 eager floating-power squares differ from NumPy,
whereas integer squares, multiplication and JIT floating squares match.
The current public quadrature result also matches this fixture. This isolates
an expression/lowering difference rather than attributing it to a particular
processor instruction. Other devices or fused calculations can still round
differently; ``quadrature_sum`` invokes the original complete statistic.

Validation procedure
----------------------

Keep data, grids, template metadata, precision and native FFT settings fixed.
Start with all relevant original controls and require exact results, then
remove one control at a time. Include event decisions and ordering in addition
to numerical error summaries. The controls are slow host validation boundaries;
the default remains JAX execution on the selected device. Conditioning, PSD
and transform comparisons require their own corresponding controls. Template
generation is held fixed in these examples.
