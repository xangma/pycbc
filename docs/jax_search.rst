.. _jax-search:

JAX batched filtering and event processing
==========================================

The JAX library interfaces batch matched filtering and keep numerical event
columns on the selected device. Use ``JAXScheme`` to select execution;
``reference_operations`` is empty by default. Conditioning and PSD preparation
also run on device by default. See :doc:`jax_arrays`, :doc:`jax_filtering` and
:doc:`jax_psd` for their interfaces and validation controls.

These are library APIs. The ordinary ``pycbc_live`` executable retains its
existing pipeline and scheduling.

Batched matched filtering
-------------------------

``pycbc.filter.matchedfilter_jax.batch_matched_filter_bank`` accepts a matrix
with shape ``(number_of_templates, number_of_frequency_bins)`` or a list of
``FrequencySeries`` objects, one data spectrum and an optional PSD. Inputs
must share a frequency grid. Supply ``delta_f`` for raw arrays; series inputs
can provide it through their metadata. The returned JAX arrays contain
normalized complex SNR time series and one template ``sigmasq`` per row.

.. code-block:: python

   import numpy as np
   from pycbc.scheme import JAXScheme
   from pycbc.filter.matchedfilter_jax import batch_matched_filter_bank

   rng = np.random.default_rng(512)
   templates = (rng.normal(size=(3, 65))
                + 1j * rng.normal(size=(3, 65))).astype(np.complex64)
   data = (rng.normal(size=65) + 1j * rng.normal(size=65)).astype(np.complex64)
   psd = np.ones(65, dtype=np.float32)
   with JAXScheme("cpu"):
       snr, sigmasq = batch_matched_filter_bank(
           templates, data, psd, delta_f=0.5,
           low_frequency_cutoff=1.3, high_frequency_cutoff=25.9)

This synthetic example illustrates the interface, rather than waveform
generation. The frequency band follows the ordinary single-template cutoff
rules, including DC and Nyquist handling. Correlation is formed before PSD
division. Complex64 inputs retain complex64 SNR storage; normalization
reductions accumulate in double precision.

``batch_peak_values`` reduces a contiguous batch over a common sample slice
and returns a peak index relative to that slice and its complex value for each
row. ``BatchCorrelator`` uses the standard correlation interface under a JAX
scheme. Its default accepts mutable template inputs; explicitly setting
``immutable_templates=True`` allows reusable packing when the templates are
unchanged.

Live filtering
---------------

``pycbc.filter.matchedfilter.LiveBatchMatchedFilter`` selects its JAX backend
when constructed and used inside ``JAXScheme``. Call ``process_data`` with a
data reader that supplies ``overwhitened_data(delta_f)``, its associated PSD,
sample rate, block size, padding and start time. Templates retain the ordinary
bank metadata and normalization contract.

Templates are grouped by their original duration and frequency spacing. JAX
does not pad all templates onto a different common frequency grid. The
``maxelements`` setting controls batch sizes within those groups.

Correlation, inverse transforms, normalization, peak selection and numerical
veto reductions use JAX by default. Python handles template metadata and
orchestration; compact selection/count results can synchronize to the host.
The result preserves the ordinary Live column names, including ``snr``,
``coa_phase``, ``end_time``, ``template_id``, ``sigmasq`` and veto columns.
SNR, phase and ``sigmasq`` storage remain float32.

Events and coincidences
------------------------

The ordinary ranking, trigger-cut, segment-veto, FindChirp and coincidence
functions dispatch to JAX for the active scheme or JAX numeric inputs.
``newsnr`` and ``effsnr`` promote calculation inputs to float64, following
their CPU implementations. Ranking getters preserve their usual float32
output columns. Segment intervals are half-open.

Use ``pycbc.events.eventmgr_jax.JAXEventManager`` explicitly for device-resident
event storage. It supports template clustering, chi-square and NewSNR cuts,
and loudest-event selection. Its output/checkpoint boundaries materialize
ordinary host data. Default JAX consolidation does not support injection-window
selection or chirp-width loudest bins; the ``event_loudest`` original control
can validate the latter selection.

Use ``pycbc.events.coinc_jax.JAXLiveCoincTimeslideBackgroundEstimator`` explicitly
for JAX Live coincidence and background state. The ordinary CPU estimator
retains its original buffer types. The standalone ``time_coincidence``,
``cluster_over_time``, ``cluster_coincs`` and ``cluster_coincs_multiifo``
interfaces retain their ordinary arguments.

Original-implementation validation
----------------------------------

Select original operations independently while other stages remain JAX:

.. code-block:: python

   with JAXScheme("cpu", reference_operations=(
           "correlate", "divide", "ifft", "weighted_inner")):
       snr, sigmasq = batch_matched_filter_bank(
           templates, data, psd, delta_f=0.5,
           low_frequency_cutoff=1.3, high_frequency_cutoff=25.9)

Native controls copy inputs to the host and return array results to the
selected device. Some invoke original routines in an isolated CPU process.
They are slow validation paths; tracing and differentiation cannot cross them.
Use the same native FFT backend and settings for an exact comparison.

.. list-table:: Search validation controls
   :header-rows: 1
   :widths: 45 55

   * - Operation names
     - Original calculation selected
   * - ``correlate``, ``divide``, ``ifft``
     - Correlation, PSD division and inverse transform
   * - ``inner``, ``weighted_inner``, ``squared_norm``
     - Unweighted/PSD-weighted reductions and complex power
   * - ``abs_arg_max``, ``live_selection``
     - Peak reduction and Live scaling/threshold/result storage
   * - ``power_chisq_bins``, ``power_chisq_at_points``
     - Power chi-square bins and selected-point calculation
   * - ``newsnr``, ``effsnr``
     - Individual ranking formulas
   * - ``segment_veto``, ``findchirp_cluster``
     - Segment selection and greedy sample clustering
   * - ``event_chisq_threshold``, ``event_newsnr_threshold``, ``event_loudest``
     - Individual original event-manager selections
   * - ``time_coincidence``, ``cluster_over_time``
     - Coincidence construction and time clustering
   * - ``cluster_coincs``, ``cluster_coincs_multiifo``
     - Mean-time/slide geometry and clustering together
   * - ``quadrature_sum``
     - Original coincident quadrature statistic

See :doc:`jax_search_numerical_differences` and its executable notebook for
controlled comparisons and composed exact results. Conditioning, PSD and veto
controls are described with their owning interfaces. Template generation is
held fixed in these examples.

.. toctree::
   :maxdepth: 1
   :hidden:

   jax_search_numerical_differences
