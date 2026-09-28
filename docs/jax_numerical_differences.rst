.. _jax-numerical-differences:

Numerical Agreement and Differences: JAX Backend vs. Pristine PyCBC CPU
=======================================================================

This document provides a concise, rigorous explanation of the numerical
relationship between the JAX acceleration backend and the pristine PyCBC CPU
reference. The companion notebook and selected receipts below preserve the
supporting configurations, source hashes, and diagnostic evidence.

An interactive companion Jupyter notebook walking through each stage comparison
step-by-step on real LIGO Hanford detector data (GW150914) is provided at
``examples/jax/jax_numerical_parity.ipynb``.

Executive Summary: Scoped Search-Outcome Agreement
--------------------------------------------------

Across the recorded 2,048 Hz, single-precision (complex64) search workloads on
real gravitational-wave detector data (LIGO Hanford H1 and Livingston L1):

1. **100% Retained Trigger Identity Match**: Every retained candidate trigger in
   both ``pycbc_inspiral`` (77/77) and ``pycbc_live`` (27/27) matches the pristine
   CPU reference exactly. Retained coalescence arrival times match to the exact sample.
2. **High-Precision SNR and Phase Agreement**: Retained signal-to-noise ratio
   (SNR) agrees within a relative difference of :math:`< 1.6 \times 10^{-5}`.
   Coalescence phase agrees within :math:`< 1.8 \times 10^{-6}` radians.
3. **Zero Changed Detection Decisions**: Audits covering over 17,000 candidate
   selection, clustering, and ranking records confirm that **zero detection
   candidate decisions, threshold cuts, or ranking orders are altered**.
4. **Synthetic Simulation Parity**: In synthetic Gaussian noise end-to-end
   simulations with hardware injections, all retained fields—including
   chi-square—pass the strict automated comparator tolerances.
5. **Localized Intermediate Differences on Detector Data**: Real detector data
   exhibits localized differences in intermediate float32 FFT spectra, Power
   Spectral Density (PSD) estimation below the 30 Hz search cutoff, and a small
   subset of retained chi-square values. Controlled experiments localize the
   observed differences to the mechanisms detailed below.

These results establish scoped agreement for the retained triggers and audited
decisions in the recorded workloads. They do **not** constitute a complete
scientific qualification: the frozen comparator still fails some chi-square
values and exact conditioned-strain, segment-spectrum, and PSD checks.

.. list-table:: Retained Search Comparison Summary (Pristine CPU vs. JAX)
   :header-rows: 1
   :widths: 28 22 22 28

   * - Metric / Field
     - Live Search (H1/L1)
     - Inspiral Search (H1)
     - Verdict
   * - Retained Trigger Identities
     - 27 / 27 (100%)
     - 77 / 77 (100%)
     - Exact match
   * - Retained End Times
     - Exact to sample
     - Exact to sample
     - Exact match
   * - Max Absolute SNR Difference
     - :math:`1.62 \times 10^{-5}`
     - :math:`4.77 \times 10^{-6}`
     - Pass (tolerance :math:`1 \times 10^{-4}`)
   * - Max Absolute Phase Difference
     - :math:`1.79 \times 10^{-6}\text{ rad}`
     - :math:`3.28 \times 10^{-7}\text{ rad}`
     - Pass (tolerance :math:`1 \times 10^{-4}\text{ rad}`)
   * - Candidate Selection Decisions
     - 0 changed decisions
     - 0 changed decisions
     - Invariant
   * - Chi-Square Values
     - 2/27 CPU, 3/27 CUDA fail
     - 4/77 CPU, 13/77 CUDA fail
     - Bounded (:math:`\sim 0.3\%\text{--}0.7\%` max)
   * - Conditioned Strain & PSD
     - Strict array check fails
     - Strict array check fails
     - Bounded numerical differences

Root Causes of Numerical Differences
------------------------------------

Detailed stage-attribution experiments, cross-feed tests, and independent
extended-precision numerical oracles identify three distinct mechanisms:

1. Chi-Square Differences: Discontinuous Bin-Edge Discretization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Power Chi-Square veto (:math:`\chi^2`) tests whether signal power is
distributed across frequencies as predicted by the template. The frequency band
is divided into :math:`p` bins of equal expected power:

.. math::

   \int_{f_{b-1}}^{f_b} \frac{|\tilde{h}(f)|^2}{S_n(f)}\,df = \frac{1}{p} \int_{f_{\text{low}}}^{f_{\text{high}}} \frac{|\tilde{h}(f)|^2}{S_n(f)}\,df

In PyCBC, ``power_chisq_bins`` computes cumulative sums of :math:`|\tilde{h}|^2 / S_n`
and finds the integer sample indices :math:`k_b` crossing the equal-power thresholds.

* **The Mechanism**: Because of sub-ppm floating-point rounding differences in the
  PSD between FFT libraries, the cumulative power near a bin boundary can differ
  by a fraction of a ULP. Because frequency bin indices are discrete integers, this
  minute continuous variation occasionally shifts an integer boundary by
  :math:`\pm 1` sample (for example, index ``88548`` instead of ``88549``).
* **The Impact**: Shifting an integer bin edge by one index includes or excludes
  an entire discrete frequency component from that bin's discrete SNR sum. This
  step change produces a :math:`\sim 0.3\%\text{--}0.7\%` shift in the resulting
  statistic.
* **Cross-Feed Evidence**: Controlled experiments—holding the input
  correlations and normalizations fixed while swapping bin edges—reproduced
  approximately the full signed chi-square difference in the tested failing
  captures (97% to 101%, depending on the case). This localizes those retained
  differences to the one-bin boundary choice; the frozen chi-square gate still
  fails. When bin edges are held identical, the JAX and CPU chi-square
  calculation kernels agree within :math:`\sim 10^{-5}`.

2. PSD (Welch) Differences: Precision and FFT Rounding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyCBC estimates the detector noise PSD using Welch's method (averaged periodograms
over windowed strain segments).

* **The Mechanism**: The pristine CPU reference evaluates the Welch forward FFT in
  single-precision (float32) using Intel MKL. The JAX backend widens the Welch FFT
  and accumulation to double precision (float64).
* **Accuracy Evidence**: For the captured Welch inputs and frequencies,
  independent numerical oracles (using 80-bit extended precision and SciPy
  float64 direct sums) place the widened JAX result closer to the oracle than
  the single-precision MKL result. In the sub-30 Hz band (below the search
  cutoff), the recorded MKL calculation has relative errors exceeding
  :math:`10^{-4}`, whereas the recorded widened JAX calculation has none above
  that threshold. This diagnostic does not override the pristine reference or
  qualify the complete search.
* **FFT Library Sensitivity**: Even on CPU, switching the pristine reference from
  Intel MKL to FFTW changes 30,637 in-band Welch PSD bins and causes the pristine
  CPU path to fail the exact same four chi-square values against itself.

.. _jax-highpass-compat-evidence:

3. Segment Spectra and Conditioning Differences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Floating-Point Non-Associativity**: Fast Fourier Transforms involve extensive
  sums of trigonometric products. Different optimized implementations (Intel MKL,
  FFTW, NVIDIA cuFFT, and XLA) group operations differently. Single-precision
  floating-point addition is non-associative (:math:`(a + b) + c \neq a + (b + c)`),
  producing ULP-level variations in complex64 Fourier coefficients.
* **High-Pass Filter Recurrence**: Pristine CPU uses LAL's time-domain IIR filter.
  The default JAX mode (``--jax-highpass-mode lal-serial``) evaluates the exact
  same sample recurrence via ``jax.lax.scan``, matching LAL down to machine
  precision. The optional parallel mode (``--jax-highpass-mode parallel``) uses an
  associative prefix scan, which achieves high GPU concurrency at the cost of
  slight operation reordering.

Stage-by-Stage Evidence Ledger
------------------------------

The following table summarizes the disposition of observed differences:

.. list-table:: Scientific Stage-by-Stage Ledger
   :header-rows: 1
   :widths: 22 48 30

   * - Stage / Array
     - Measured Evidence
     - Scientific Disposition
   * - Retained Triggers & Arrival Times
     - 100% identity and sample-exact arrival time match in the recorded H1 and
       L1 detector-search workloads.
     - **Scoped pass** for the retained records in these workloads.
   * - Retained SNR & Phase
     - Relative SNR differences :math:`< 1.6 \times 10^{-5}`; phase differences
       :math:`< 1.8 \times 10^{-6}\text{ rad}`.
     - **Scoped pass** under the stated field tolerances.
   * - Candidate Decisions & Ranking
     - Audited 17,000+ candidate selection records across template clustering,
       NewSNR thresholds, and loudest-event retention; zero decisions changed.
     - **Scoped pass** for the recorded decision audit.
   * - Welch PSD Estimation
     - JAX widens Welch FFT to float64. Independent 80-bit direct sums confirm JAX
       has lower RMS relative error than MKL float32. Sub-30 Hz differences have
       zero impact on search band (:math:`\ge 30\text{ Hz}`).
     - **Diagnostic finding** for the captured inputs; the frozen full-PSD gate
       remains failed.
   * - Segment Fourier Spectra
     - MKL vs. cuFFT/XLA float32 transform rounding produces differences of
       :math:`\sim 2 \times 10^{-7}` relative L2 norm.
     - Cross-library controls reproduce the sensitivity, but the exact frozen
       segment-spectrum gate remains failed.
   * - Retained Chi-Square Values
     - :math:`\sim 0.3\%\text{--}0.7\%` relative difference on a small fraction of
       triggers; 97%--101% explained by :math:`\pm 1` bin-edge shifts in
       ``power_chisq_bins``. Kernels agree within :math:`\sim 10^{-5}` at fixed edges.
     - Cross-feed controls explain the tested captures, but the frozen
       chi-square gate remains failed.
   * - Conditioned Strain (High-Pass)
     - ``lal-serial`` reproduces LAL sample recurrence. Parallel prefix mode
       introduces minor reordering differences.
     - ``lal-serial`` preserves the tested recurrence; complete-search
       conditioned-array gates must still pass independently.

Rigorous Testing and Verification Framework
-------------------------------------------

To ensure scientific validity and guard against regressions, the JAX backend is
verified through a multi-tiered testing framework:

1. **Unit & Integration Suites** covering JAX modules, including:

   * Array semantics, device dispatch, and DLPack buffer zero-copy contracts.
   * Batch matched filtering, time-slice cropping, and peak clustering
     (``test/test_jax_search.py``).
   * Frequency-domain and time-domain waveform generation and decompression
     (``test/waveform/test_jax_*.py``).
   * Pure-JAX analytical PSD models, Welch estimation, and interpolation
     (``test/test_jax_psd.py``).
   * Event manager, coincidences, and ranking statistics
     (``test/test_jax_eventmgr.py``, ``test/test_jax_coinc.py``).

2. **Science Regressions and Numerical Oracles**:

   * Direct double-precision Fourier sum oracles testing long-segment point
     power without prefix subtraction (``test/test_jax_chisq_science.py``).
   * CPU reference and zero-padded crop FFT parity tests
     (``test/test_jax_cpu_chisq.py``, ``test/test_chisq_numpy.py``).
   * Dual-mode chi-square verification comparing ``cpu-compatible`` and
     ``direct-phase`` modes (``test/test_jax_chisq_modes.py``).

3. **End-to-End Simulation Benchmarks**:

   * Synthetic Gaussian noise injection campaigns testing complete pipeline
     execution against pristine CPU reference outputs (``test/test_benchmark_science.py``).

4. **Decision-Level Auditing**:

   * Verification of template clustering, thresholding, and event ranking invariance
     across pipeline executions (``test/test_benchmark_decisions.py``).

.. _jax-benchmark-protocol:

Benchmark Qualification Protocol and Numeric Rules
--------------------------------------------------

To evaluate full-search executables (``pycbc_inspiral`` and ``pycbc_live``), the
qualification harness in ``tools/benchmark_science.py`` compares candidate runs
against a pinned, pristine CPU reference across four execution arms: ``cpu``
(pristine upstream reference), ``branch_cpu`` (candidate checkout on CPU),
``jax_cpu`` (``jax:cpu``), and ``jax_cuda`` (``jax:cuda:0``).

The workload fixes 2,048 Hz sampling and complex64 matched-filter arrays on
identical detector frame and template bank inputs.

The maintained comparator enforces the following frozen numerical rules:

* **Trigger Identity and Timing**: Retained trigger identities, detector geometry,
  degrees of freedom, and scientific metadata must match exactly; missing evidence
  fails closed. Retained coalescence arrival times must agree to the exact sample.
* **SNR and Scientific Observables**: Standard numerical fields use ``atol=1e-5``
  and ``rtol=1e-4`` against the pristine reference. Template sensitivity (``sigmasq``)
  uses ``rtol=1e-5``.
* **Coalescence Phase**: Evaluated via circular distance on the torus with a strict
  tolerance of :math:`1 \times 10^{-4}` radians.
* **PSD Arrays**: Comparison requires ``rtol=1e-4`` and ``atol=0`` across all active
  analysis bands, handling out-of-band infinities consistently.
* **Decision-Level Invariance**: A separate gate (``tools/benchmark_decisions.py``)
  validates that intermediate candidate selection windows, threshold crossings,
  clustering choices, and final ranked candidate order are strictly invariant.

Conclusion and Qualification Status
-----------------------------------

The retained trigger identities, timing, SNR, phase, and audited ranking decisions
agree within the recorded scope. Diagnostic controls explain the observed scale
and location of several intermediate differences. The JAX backend is nevertheless
**not fully qualified under the frozen complete-search comparator**: some retained
chi-square values and exact conditioned-strain, segment-spectrum, and PSD gates
remain failed. The benchmark harness may therefore collect only explicitly
labelled **known-divergence diagnostic timings**: it preserves every failed gate
and does not treat those measurements as equivalent-output speedup claims.
Scientific qualification and publishable speedup claims still require a fresh
qualification run that passes the required gates.

An interactive Jupyter notebook walking through these stage comparisons
step-by-step using real LIGO Hanford open data (GW150914) is provided at
``examples/jax/jax_numerical_parity.ipynb``.

Primary Receipt Archives
~~~~~~~~~~~~~~~~~~~~~~~~

The following archived receipts document the empirical evidence:

* :download:`Wide-PSD search receipt <_static/jax_path_diagnosis/native-jax-wide-psd-20260923.json.gz>`
* :download:`H1/L1 conditioning controls <_static/jax_path_diagnosis/native-jax-heldout-highpass-20260923.json.gz>`
* :download:`PSD stage attribution <_static/jax_path_diagnosis/native-jax-psd-stage-attribution-20260923.json.gz>`
* :download:`Independent numerical oracles <_static/jax_path_diagnosis/native-jax-independent-oracles-20260923.json.gz>`
* :download:`H1 causal replay <_static/jax_path_diagnosis/jax-0112-h1-causal-replay-20260923.json.gz>`
* :download:`L1 edge-attribution receipt <_static/jax_path_diagnosis/jax-0112-l1-chisq-edge-attribution-20260923.json.gz>`
* :download:`Event manager decision audit <_static/jax_path_diagnosis/native-jax-event-manager-decision-audit-20260923.json.gz>`
* :download:`Maintained decision gate replay <_static/jax_path_diagnosis/native-jax-decision-gate-20260923.json.gz>`
* :download:`Targeted correction screen <_static/jax_path_diagnosis/native-jax-targeted-correction-screen-20260923.json.gz>`
